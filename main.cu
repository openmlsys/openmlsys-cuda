#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <Eigen/Core>
#include <argparse/argparse.hpp>
#include <ctime>
#include <iostream>
#include <utility>

void gemmFinal(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K);

void gemmUse128(const float *deviceAPtr, const float *deviceBPtr,
                float *deviceCPtr, float alpha, float beta, unsigned M,
                unsigned N, unsigned K);

void gemmUseTile(const float *deviceAPtr, const float *deviceBPtr,
                 float *deviceCPtr, float alpha, float beta, unsigned M,
                 unsigned N, unsigned K);

void gemmNaive(const float *deviceAPtr, const float *deviceBPtr,
               float *deviceCPtr, float alpha, float beta, unsigned M,
               unsigned N, unsigned K);

void gemmHideSmemLatency(const float *deviceAPtr, const float *deviceBPtr,
                         float *deviceCPtr, float alpha, float beta, unsigned M,
                         unsigned N, unsigned K);

void gemmTransposeSmem(const float *deviceAPtr, const float *deviceBPtr,
                       float *deviceCPtr, float alpha, float beta, unsigned M,
                       unsigned N, unsigned K);

void gemmUseSmem(const float *deviceAPtr, const float *deviceBPtr,
                 float *deviceCPtr, float alpha, float beta, unsigned M,
                 unsigned N, unsigned K);

class GemmTester {
  class cuTimer {
    cudaEvent_t startEvent{}, stopEvent{};

  public:
    cuTimer() {
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);
    }
    ~cuTimer() {
      cudaEventDestroy(stopEvent);
      cudaEventDestroy(startEvent);
    }

    void start() { cudaEventRecord(startEvent); }

    float end() {
      cudaEventRecord(stopEvent);
      auto error = cudaEventSynchronize(stopEvent);
      if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
      }
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

      return milliseconds;
    }
  };

  cuTimer timer{};
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hostC;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      deviceCCopied;
  const float *deviceAPtr, *deviceBPtr;
  float *deviceCPtr;
  const float *deviceCInitPtr;
  float alpha, beta;
  unsigned M, N, K;
  int iteration;

  void tearUp() {
    cudaMemcpy(deviceCPtr, deviceCInitPtr, M * N * sizeof(float),
               cudaMemcpyDeviceToDevice);
  }

  void checkValue() const {
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
        (hostC - deviceCCopied).array().abs();

    printf("Max Error: %f\n", diffArray.maxCoeff());
  }

  template <typename Function> void profile(Function &&gemmFunction) {
    double elapsedTime = 0;
    for (int i = 0; i < iteration; ++i) {
      tearUp();
      timer.start();
      gemmFunction(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
      elapsedTime += timer.end();
    }
    elapsedTime /= iteration;
    double GFLOPS = 2 * 1e-9 * M * N * K / (elapsedTime * 1e-3);
    printf("Average Time: %.3f ms, Throughput: %.3f GFLOPS\n", elapsedTime,
           GFLOPS);
  }

public:
  explicit GemmTester(float alpha, float beta, unsigned M, unsigned N,
                      unsigned K, int iteration)
      : hostC{M, N}, deviceCCopied{M, N}, alpha(alpha), beta(beta), M(M), N(N),
        K(K), iteration{iteration} {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M,
                                                                            K};
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B{K,
                                                                            N};
    A.setRandom();
    B.setRandom();
    hostC.setRandom();

    float *_deviceCPtr, *_deviceCInitPtr;
    cudaMalloc(&_deviceCPtr, M * N * sizeof(float));
    cudaMalloc(&_deviceCInitPtr, M * N * sizeof(float));
    deviceCPtr = _deviceCPtr;
    deviceCInitPtr = _deviceCInitPtr;
    cudaMemcpy(_deviceCInitPtr, hostC.data(), M * N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    clock_t begin, end;
    begin = clock();
    hostC = alpha * (A * B) + beta * hostC;
    end = clock();
    printf("CPU use: %.3f ms\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);

    float *_deviceAPtr, *_deviceBPtr;
    cudaMalloc(&_deviceAPtr, M * K * sizeof(float));
    cudaMalloc(&_deviceBPtr, K * N * sizeof(float));
    cudaMemcpy(_deviceAPtr, A.data(), M * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(_deviceBPtr, B.data(), K * N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    deviceAPtr = _deviceAPtr;
    deviceBPtr = _deviceBPtr;
  }
  ~GemmTester() {
    cudaFree((void *)deviceAPtr);
    cudaFree((void *)deviceBPtr);
    cudaFree(deviceCPtr);
  }

  template <typename Function>
  void evaluate(Function &&gemmFunction, const char *name) {
    tearUp();
    printf("-----------------------------------\n");
    printf("Evaluating %s\n", name);
    gemmFunction(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
    cudaMemcpy(deviceCCopied.data(), deviceCPtr, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkValue();
    profile(std::forward<Function>(gemmFunction));
    printf("-----------------------------------\n");
  }
};

class gemmCuBlas {
  cublasHandle_t handle{nullptr};

public:
  gemmCuBlas() { cublasCreate(&handle); }
  ~gemmCuBlas() { cublasDestroy(handle); }

  void operator()(const float *A, const float *B, float *C, float &alpha,
                  float &beta, int M, int N, int K) const {
    int lda = N, ldb = K, ldc = N;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, lda, A,
                ldb, &beta, C, ldc);
  }
};

int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
  case 2: // Fermi
    if (devProp.minor == 1)
      cores = mp * 48;
    else
      cores = mp * 32;
    break;
  case 3: // Kepler
    cores = mp * 192;
    break;
  case 5: // Maxwell
    cores = mp * 128;
    break;
  case 6: // Pascal
    if ((devProp.minor == 1) || (devProp.minor == 2))
      cores = mp * 128;
    else if (devProp.minor == 0)
      cores = mp * 64;
    else
      throw std::runtime_error("Unknown device type");
    break;
  case 7: // Volta and Turing
    if ((devProp.minor == 0) || (devProp.minor == 5))
      cores = mp * 64;
    else
      throw std::runtime_error("Unknown device type");
    break;
  case 8: // Ampere
    if (devProp.minor == 0)
      cores = mp * 64;
    else if (devProp.minor == 6)
      cores = mp * 128;
    else
      throw std::runtime_error("Unknown device type");
  default:
    throw std::runtime_error("Unknown device type");
  }
  return cores;
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("gemm");

  program.add_argument("--cpu_procs")
      .help("processor num used of CPU")
      .scan<'i', int>()
      .default_value(omp_get_num_procs());

  program.add_argument("--gpu_rank")
      .help("the used GPU rank")
      .scan<'i', int>()
      .default_value(0);

  program.add_argument("--repeat_iterations")
      .help("repeat iteration numbers and average the result")
      .scan<'i', int>()
      .default_value(10);

  program.add_argument("--alpha").scan<'g', float>();

  program.add_argument("--beta").scan<'g', float>();

  program.add_argument("-M").scan<'i', unsigned>();

  program.add_argument("-N").scan<'i', unsigned>();

  program.add_argument("-K").scan<'i', unsigned>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  const int cpu_procs = program.get<int>("--cpu_procs");
  const int gpu_rank = program.get<int>("--gpu_rank");
  const int repeat_iterations = program.get<int>("--repeat_iterations");
  const auto alpha = program.get<float>("--alpha");
  const auto beta = program.get<float>("--beta");
  const auto M = program.get<unsigned>("-M");
  const auto N = program.get<unsigned>("-N");
  const auto K = program.get<unsigned>("-K");
  printf("Program start with %d CPU processes on the %d-th GPU\n", cpu_procs,
         gpu_rank);
  omp_set_num_threads(cpu_procs);
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, gpu_rank);
  cudaSetDevice(gpu_rank);
  printf("GPU %s status: ", deviceProp.name);
  double boostFrequency = deviceProp.clockRate / 1e6;
  int fp32CoresNum = getSPcores(deviceProp);
  double peakPerformance = boostFrequency * fp32CoresNum * 2;
  printf("clock rate %.3f GHz, FP32 cores num %d, FP32 peak throughput %.3f "
         "GFLOPS\n",
         boostFrequency, fp32CoresNum, peakPerformance);
  printf("A: %d x %d, B: %d x %d, C: %d x %d\n", M, K, K, N, M, N);

  GemmTester tester{alpha, beta, M, N, K, repeat_iterations};
  tester.evaluate(gemmCuBlas{}, "cuBlas");
  tester.evaluate(gemmNaive, "Naive");
  tester.evaluate(gemmUse128, "Use128");
  tester.evaluate(gemmUseTile, "UseTile");
  tester.evaluate(gemmUseSmem, "UseSmem");
  tester.evaluate(gemmTransposeSmem, "TransposeSmem");
  tester.evaluate(gemmHideSmemLatency, "HideSmemLatency");
  tester.evaluate(gemmFinal, "Final");

  return 0;
}
