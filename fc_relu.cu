#include <gflags/gflags.h>
#include <omp.h>
#include <cuda_runtime_api.h>

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

DEFINE_int32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_int32(in_dim, 512, "input dim of FC");
DEFINE_int32(out_dim, 1024, "output dim of FC");
DEFINE_int32(batch_size, 128, "batch size");

int main(int argc, char *argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  const int outDim = FLAGS_out_dim;
  const int batchSize = FLAGS_batch_size;
  const int inDim = FLAGS_in_dim;

  printf(
      "Starting the problem with batch size: %d, input dim: %d, output dim: "
      "%d\n",
      batchSize, inDim, outDim);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;

  using RowMajor = cutlass::layout::RowMajor;

  using OperatorClass = cutlass::arch::OpClassSimt;
  using ArchTag = cutlass::arch::Sm50;

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

  omp_set_num_threads(FLAGS_cpu_procs);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x,
      weight, outEigen, outCUTLASS;
  Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> bias;
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
                       {outDevPtr, outDim}, {});
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
