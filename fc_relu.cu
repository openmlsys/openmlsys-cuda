#include <omp.h>

#include <Eigen/Core>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

template <typename ElementOutput_, typename ElementComputeEpilogue_,
          typename ElementAccumulator_>
struct ReLUEpilogue {
  constexpr static int kCount = 1;
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementComputeEpilogue_, kCount>;

  struct Params {};

  CUTLASS_DEVICE
  explicit ReLUEpilogue(const Params &) {}

  CUTLASS_DEVICE
  constexpr static bool is_source_needed() { return true; }

  CUTLASS_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_DEVICE
  FragmentOutput operator()(const FragmentAccumulator &) const {
    return FragmentOutput{};
  }

  CUTLASS_DEVICE
  FragmentOutput operator()(
      const FragmentCompute &fragmentCompute,
      const FragmentAccumulator &fragmentAccumulator) const {
    FragmentOutput output;
#pragma unroll
    for (unsigned i = 0; i < kCount; ++i) {
      output[i] =
          ::max(ElementOutput(0), fragmentCompute[i] + fragmentAccumulator[i]);
    }
    return output;
  }
};

const int outDim = 1024;
const int batchSize = 1024;
const int inDim = 1024;

int main() {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;

  using RowMajor = cutlass::layout::RowMajor;

  using OperatorClass = cutlass::arch::OpClassSimt;
  using ArchTag = cutlass::arch::Sm80;

  using DefaultGemmConfiguration =
      cutlass::gemm::device::DefaultGemmConfiguration<
          OperatorClass, ArchTag, ElementInputA, ElementInputB,
          ElementComputeEpilogue, ElementAccumulator>;

  using ThreadblockShape = DefaultGemmConfiguration::ThreadblockShape;
  using WarpShape = DefaultGemmConfiguration::WarpShape;
  using InstructionShape = DefaultGemmConfiguration::InstructionShape;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, RowMajor, ElementInputB, RowMajor, ElementOutput, RowMajor,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
      InstructionShape,
      ReLUEpilogue<ElementOutput, ElementComputeEpilogue, ElementAccumulator>>;

  omp_set_num_threads(omp_get_num_procs());

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x,
      weight, outEigen, outCUTLASS;
  Eigen::Matrix<float, 1, Eigen::Dynamic> bias;
  weight.resize(inDim, outDim);
  x.resize(batchSize, inDim);
  bias.resize(outDim);

  weight.setRandom();
  x.setRandom();
  bias.setRandom();
  outEigen = ((x * weight).array().rowwise() + bias.array()).cwiseMax(0);
  outCUTLASS.resize(outEigen.rows(), outEigen.cols());

  float *xDevPtr, *weightDevPtr, *biasDevPtr, *outDevPtr;
  cudaMalloc(&xDevPtr, x.size() * sizeof(float));
  cudaMemcpy(xDevPtr, x.data(), x.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&weightDevPtr, weight.size() * sizeof(float));
  cudaMemcpy(weightDevPtr, weight.data(), weight.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&biasDevPtr, bias.size() * sizeof(float));
  cudaMemcpy(biasDevPtr, bias.data(), bias.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&outDevPtr, outEigen.size() * sizeof(float));

  Gemm::Arguments args({batchSize, outDim, inDim}, {xDevPtr, inDim},
                       {weightDevPtr, outDim}, {biasDevPtr, 0},
                       {outDevPtr, inDim}, {});
  Gemm gemm_op;
  gemm_op(args);
  cudaDeviceSynchronize();
  cudaMemcpy(outCUTLASS.data(), outDevPtr, outCUTLASS.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("Max error: %f\n", (outEigen - outCUTLASS).cwiseAbs().maxCoeff());
  cudaFree(xDevPtr);
  cudaFree(weightDevPtr);
  cudaFree(biasDevPtr);
  cudaFree(outDevPtr);
  return 0;
}
