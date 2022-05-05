#include "util.cuh"

namespace {
template <typename LayoutTile, typename LayoutBlock, typename LayoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  constexpr unsigned ratio = sizeof(openmlsys::float4) / sizeof(float);
  unsigned int m = threadIdx.x * LayoutThread::m + LayoutTile::m * blockIdx.x;
  unsigned int n = threadIdx.y * LayoutThread::n + LayoutTile::n * blockIdx.y;
  openmlsys::Tensor2D<const float> pA{A, M, K};
  pA.addOffset(m, 0);
  openmlsys::Tensor2D<const openmlsys::float4> pB{B, K, N / ratio};
  pB.addOffset(0, n / ratio);
  openmlsys::Tensor2D<openmlsys::float4> pC{C, M, N / ratio};
  pC.addOffset(m, n / ratio);
  if (!pC.validOffset(0, 0)) return;

  const unsigned iterationA = LayoutTile::m / LayoutBlock::m / LayoutThread::m;
  const unsigned iterationB = LayoutTile::n / LayoutBlock::n / LayoutThread::n;
  const unsigned intervalA = LayoutTile::m / iterationA;
  const unsigned intervalB = LayoutTile::n / iterationB;
  openmlsys::float4 c[iterationA][iterationB][4];
  memset(c, 0, sizeof(c));
  for (unsigned k = 0; k < K; ++k) {
#pragma unroll
    for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
#pragma unroll
      for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
        openmlsys::float4 fragmentA{};
#pragma unroll
        for (unsigned i = 0; i < ratio; ++i) {
          fragmentA[i] = pA(i + iterA * intervalA, k);
        }
        openmlsys::float4 fragmentB = pB(k, iterB * intervalB / ratio);

#pragma unroll
        for (unsigned i = 0; i < ratio; ++i) {
          c[iterA][iterB][i] = c[iterA][iterB][i] + fragmentB * fragmentA[i];
        }
      }
    }
  }

#pragma unroll
  for (auto &termA : c) {
#pragma unroll
    for (auto &termB : termA) {
#pragma unroll
      for (auto &term : termB) {
        term = term * alpha;
      }
    }
  }

#pragma unroll
  for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
#pragma unroll
    for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
#pragma unroll
      for (unsigned i = 0; i < ratio; ++i) {
        openmlsys::float4 result{c[iterA][iterB][i]};
        if (beta != 0) {
          result = result +
                   pC(i + iterA * intervalA, iterB * intervalB / ratio) * beta;
        }
        pC(i + iterA * intervalA, iterB * intervalB / ratio) = result;
      }
    }
  }
}
}  // namespace

void gemmUseTile(const float *deviceAPtr, const float *deviceBPtr,
                 float *deviceCPtr, float alpha, float beta, unsigned M,
                 unsigned N, unsigned K) {
  using LayoutTile = openmlsys::Layout<128, 128, 16>;
  using LayoutBlock = openmlsys::Layout<16, 16>;
  using LayoutThread = openmlsys::Layout<4, 4>;

  dim3 block(LayoutBlock::m, LayoutBlock::n);
  dim3 grid((M * LayoutTile::m / LayoutBlock::m - 1) / LayoutBlock::m + 1,
            (N * LayoutTile::n / LayoutBlock::n - 1) / LayoutBlock::n + 1);

  gemmKernel<LayoutTile, LayoutBlock, LayoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}
