#include "util.cuh"

namespace {
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
  float c = 0;
  openmlsys::Tensor2D<const float> pA{A, M, K};
  openmlsys::Tensor2D<const float> pB{B, K, N};
  openmlsys::Tensor2D<float> pC{C, M, N};
  if (!pC.validOffset(m, n)) return;
  for (unsigned k = 0; k < K; ++k) {
    c += pA(m, k) * pB(k, n);
  }
  c = c * alpha;
  float result = c;
  if (beta != 0) {
    result = result + pC(m, n) * beta;
  }
  pC(m, n) = result;
}
}  // namespace

void gemmNaive(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  gemmKernel<<<grid, block>>>(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta,
                              M, N, K);
}
