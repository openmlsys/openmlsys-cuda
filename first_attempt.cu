#include <Eigen/Core>

__global__ void gemmKernel(const float * A,
                           const float * B, float * C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
  unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
  if (m >= M || n >= N) return;
  float c = 0;
  for (unsigned k = 0; k < K; ++k) {
    c += A[m * K + k] * B[k * N + n];
  }
  c = c * alpha;
  float result = c;
  if (beta != 0) {
    result = result + C[m * N + n] * beta;
  }
  C[m * N + n] = result;
}

void gemmNaive(const float *A, const float *B, float *C,
               float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
  dim3 block(32, 32);
  dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

  gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

using namespace Eigen;

int main() {
  unsigned M = 1024, N = 1024, K = 1024;
  float alpha = 1., beta = 0.;
  float *deviceAPrt, *deviceBPtr, *deviceCPtr;
  Matrix<float, Dynamic, Dynamic, RowMajor> A{M, K}, B{K, N}, C{M, N};
  A.setRandom();
  B.setRandom();
  C.setRandom();
  cudaMalloc(&deviceAPrt, M * K * sizeof(float));
  cudaMemcpy(deviceAPrt, A.data(), M * K * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&deviceBPtr, K * N * sizeof(float));
  cudaMemcpy(deviceBPtr, B.data(), K * N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&deviceCPtr, M * N * sizeof(float));
  cudaMemcpy(deviceCPtr, C.data(), M * N * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent);
  gemmNaive(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
  cudaEventRecord(stopEvent);

  cudaEventSynchronize(stopEvent);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
  printf("Elapsed Time: %.3f(ms)\n", milliseconds);

  cudaEventDestroy(stopEvent);
  cudaEventDestroy(startEvent);

  Matrix<float, Dynamic, Dynamic, RowMajor> hostResult{M, N}, deviceResult{M, N};
  hostResult = alpha * (A * B) + beta * C;
  cudaMemcpy(deviceResult.data(), deviceCPtr, M * N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
      (hostResult - deviceResult).array().abs();
  printf("Max Error: %f\n", diffArray.maxCoeff());
}
