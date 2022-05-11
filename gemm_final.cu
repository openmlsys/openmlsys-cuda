#include "util.cuh"

namespace {
template <typename LayoutTile, typename LayoutBlock, typename LayoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  constexpr unsigned ratio = sizeof(openmlsys::float4) / sizeof(float);
  using LayoutTileT =
      openmlsys::Layout<LayoutTile::m / ratio, LayoutTile::n / ratio,
                                LayoutTile::k / ratio>;
  using LayoutThreadT =
      openmlsys::Layout<LayoutThread::m / ratio, LayoutThread::n / ratio>;
  constexpr unsigned blockSize = LayoutBlock::m * LayoutBlock::n;
  constexpr openmlsys::float4 float4Zero{0.f, 0.f, 0.f, 0.f};

  __shared__ openmlsys::float4 tileA[2][LayoutTile::k][LayoutTileT::m];
  __shared__ openmlsys::float4 tileB[2][LayoutTile::k][LayoutTileT::n];

  const unsigned nInTileC = threadIdx.x % LayoutBlock::m;
  const unsigned mInTileC = threadIdx.x / LayoutBlock::m;

  const unsigned kInTileA = threadIdx.x % LayoutTileT::k;
  const unsigned mInTileA = threadIdx.x / LayoutTileT::k;

  const unsigned nInTileB = threadIdx.x % LayoutTileT::n;
  const unsigned kinTileB = threadIdx.x / LayoutTileT::n;

  openmlsys::Tensor2D<const openmlsys::float4> pA{A, M, K / ratio};
  pA.addOffset(LayoutTile::m * blockIdx.y + mInTileA, kInTileA);
  openmlsys::Tensor2D<const openmlsys::float4> pB{B, K, N / ratio};
  pB.addOffset(kinTileB,
               LayoutTileT::n * blockIdx.x + nInTileB * LayoutThreadT::n);
  openmlsys::Tensor2D<openmlsys::float4> pC{C, M, N / ratio};
  pC.addOffset(LayoutTile::m * blockIdx.y + mInTileC * LayoutThread::m,
               LayoutTileT::n * blockIdx.x + nInTileC * LayoutThreadT::n);

  constexpr unsigned tileSizeA = LayoutTile::m * LayoutTile::k;
  constexpr unsigned tileSizeB = LayoutTile::n * LayoutTile::k;
  constexpr unsigned tileIterationsA = tileSizeA / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalA = blockSize / LayoutTileT::k;
  constexpr unsigned tileSharedIntervalAT = LayoutTileT::m / tileIterationsA;
  constexpr unsigned tileIterationsB = tileSizeB / blockSize / ratio;
  constexpr unsigned tileGlobalIntervalB = blockSize / LayoutTileT::n;
  constexpr unsigned tileSharedIntervalBT = LayoutTileT::n / tileIterationsB;

  openmlsys::float4 bufferA[tileIterationsA];
  openmlsys::float4 bufferB[tileIterationsB];
  bool validLoadTileA[tileIterationsA];
  bool validLoadTileB[tileIterationsB];

#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
    validLoadTileA[i] =
        pA.validRowOffset(i * tileGlobalIntervalA) && pA.validColOffset(0);
    bufferA[i] =
        validLoadTileA[i] ? pA(i * tileGlobalIntervalA, 0) : float4Zero;
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    validLoadTileB[i] =
        pB.validColOffset(0) && pB.validRowOffset(i * tileGlobalIntervalB);
    bufferB[i] =
        validLoadTileB[i] ? pB(i * tileGlobalIntervalB, 0) : float4Zero;
  }

  openmlsys::float4 c[tileIterationsA * LayoutThread::m]
             [tileIterationsB * LayoutThreadT::n];
  memset(c, 0, sizeof(c));
  bool writeStageIdx = false;
#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
#pragma unroll
    for (unsigned j = 0; j < LayoutThread::m; ++j) {
      tileA[writeStageIdx][kInTileA * ratio + j][i * tileSharedIntervalAT]
           [mInTileA] = bufferA[i][j];
    }
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    tileB[writeStageIdx][kinTileB + i * tileGlobalIntervalB][nInTileB] =
        bufferB[i];
  }

  writeStageIdx = !writeStageIdx;

  __syncthreads();

  openmlsys::float4 fragmentA[2][tileIterationsA * LayoutThreadT::m];
  openmlsys::float4 fragmentB[2][tileIterationsB * LayoutThreadT::n];

#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
    fragmentA[0][i] =
        tileA[!writeStageIdx][0][i * tileSharedIntervalAT + mInTileC];
  }
#pragma unroll
  for (unsigned i = 0; i < tileIterationsB; ++i) {
    fragmentB[0][i] =
        tileB[!writeStageIdx][0][i * tileSharedIntervalBT + nInTileC];
  }

  for (unsigned i = LayoutTile::k; i < K + LayoutTile::k; i += LayoutTile::k) {
    pA.addOffset(0, LayoutTileT::k);
    pB.addOffset(LayoutTile::k, 0);
#pragma unroll
    for (unsigned j = 0; j < tileIterationsA; ++j) {
      validLoadTileA[j] = validLoadTileA[j] && pA.validColOffset(0);
      bufferA[j] =
          validLoadTileA[j] ? pA(j * tileGlobalIntervalA, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < tileIterationsB; ++j) {
      validLoadTileB[j] =
          validLoadTileB[j] && pB.validRowOffset(j * tileGlobalIntervalB);
      bufferB[j] =
          validLoadTileB[j] ? pB(j * tileGlobalIntervalB, 0) : float4Zero;
    }

#pragma unroll
    for (unsigned j = 0; j < LayoutTile::k; j++) {
      if ((i < K) && (j == LayoutTile::k - 1)) {
#pragma unroll
        for (unsigned d = 0; d < tileIterationsA; ++d) {
#pragma unroll
          for (unsigned e = 0; e < LayoutThread::m; ++e) {
            tileA[writeStageIdx][kInTileA * ratio + e][d * tileSharedIntervalAT]
                 [mInTileA] = bufferA[d][e];
          }
        }
#pragma unroll
        for (unsigned a = 0; a < tileIterationsB; ++a) {
          tileB[writeStageIdx][kinTileB + a * tileGlobalIntervalB][nInTileB] =
              bufferB[a];
        }
        writeStageIdx = !writeStageIdx;
        __syncthreads();
      }
#pragma unroll
      for (unsigned a = 0; a < tileIterationsA; ++a) {
        fragmentA[(j + 1) % 2][a] =
            tileA[!writeStageIdx][(j + 1) % LayoutTile::k][a * tileSharedIntervalAT + mInTileC];
      }
#pragma unroll
      for (unsigned a = 0; a < tileIterationsB; ++a) {
        fragmentB[(j + 1) % 2][a] =
            tileB[!writeStageIdx][(j + 1) % LayoutTile::k][a * tileSharedIntervalBT + nInTileC];
      }
#pragma unroll
      for (unsigned d = 0; d < tileIterationsA * LayoutThread::m; ++d) {
#pragma unroll
        for (unsigned e = 0; e < tileIterationsB * LayoutThreadT::n; ++e) {
          c[d][e] =
              c[d][e] +
              fragmentB[j % 2][e] *
                  fragmentA[j % 2][d / LayoutThread::m][d % LayoutThread::m];
        }
      }
    }
  }

#pragma unroll
  for (auto &a : c) {
#pragma unroll
    for (auto &b : a) {
      b = b * alpha;
    }
  }

#pragma unroll
  for (unsigned i = 0; i < tileIterationsA; ++i) {
#pragma unroll
    for (unsigned a = 0; a < LayoutThread::m; a++) {
      const bool mValid = pC.validRowOffset(a);
#pragma unroll
      for (unsigned b = 0; b < tileIterationsB; b++) {
        const bool nValid = pC.validColOffset(b * tileSharedIntervalBT);
        if (mValid && nValid) {
          openmlsys::float4 result{c[a + i * LayoutThread::m][b]};
          if (beta != 0) {
            result = result + pC(a, b * tileSharedIntervalBT) * beta;
          }
          pC(a, b * tileSharedIntervalBT) = result;
        }
      }
    }
    pC.addOffset(tileSharedIntervalAT * ratio, 0);
  }
}
}  // namespace

void gemmFinal(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
  using LayoutTile = openmlsys::Layout<128, 128, 16>;
  using LayoutBlock = openmlsys::Layout<16, 16>;
  using LayoutThread = openmlsys::Layout<4, 4>;

  dim3 block(LayoutBlock::m * LayoutBlock::n);
  dim3 grid((M - 1) / LayoutTile::m + 1, (N - 1) / LayoutTile::n + 1);

  gemmKernel<LayoutTile, LayoutBlock, LayoutThread><<<grid, block>>>(
      deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}
