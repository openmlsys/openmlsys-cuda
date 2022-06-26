#include "util.cuh"

namespace {
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  constexpr unsigned ratio = sizeof(openmlsys::float4) / sizeof(float);
  unsigned int m = (threadIdx.x + blockDim.x * blockIdx.x) * ratio;
  unsigned int n = (threadIdx.y + blockDim.y * blockIdx.y) * ratio;
  openmlsys::Tensor2D<const float> pA{A, M, K};
  pA.addOffset(m, 0);
  openmlsys::Tensor2D<const openmlsys::float4> pB{B, K, N / ratio};
  pB.addOffset(0, n / ratio);
  openmlsys::Tensor2D<openmlsys::float4> pC{C, M, N / ratio};
  pC.addOffset(m, n / ratio);
  if (!pC.validOffset(0, 0)) return;

  openmlsys::float4 c[4];
  memset(c, 0, sizeof(c));
  for (unsigned k = 0; k < K; ++k) {
    openmlsys::float4 fragmentA{};
#pragma unroll
    for (unsigned i = 0; i < ratio; ++i) {
      fragmentA[i] = pA(i, k);
    }
    openmlsys::float4 fragmentB = pB(k, 0);

#pragma unroll
    for (unsigned i = 0; i < ratio; ++i) {
      c[i] = c[i] + fragmentB * fragmentA[i];
    }
  }

#pragma unroll
  for (auto &a : c) {
    a = a * alpha;
  }

#pragma unroll
  for (unsigned i = 0; i < ratio; ++i) {
    openmlsys::float4 result = c[i];
    if (beta != 0) {
      result = c[i] + pC(i, 0) * beta;
    }
    pC(i, 0) = result;
  }
}
}  // namespace

void gemmUse128(const float *deviceAPtr, const float *deviceBPtr,
                float *deviceCPtr, float alpha, float beta, unsigned M,
                unsigned N, unsigned K) {
  dim3 block(16, 16);
  dim3 grid((M / 4 - 1) / block.x + 1, (N / 4 - 1) / block.y + 1);

  gemmKernel<<<grid, block>>>(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta,
                              M, N, K);
}
