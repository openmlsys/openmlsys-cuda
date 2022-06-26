# openmlsys-cuda

## Dependencies

- Eigen: CPU linear algebra template library
- OpenMP: Enable multi-threads acceleration on CPU
- CUDA toolkit: Compile GPU kernels and analyse GPU executions
- CUTLASS: GPU GEMM template library

### Installation Hints

- Eigen: Use package manager, e.g. `apt install libeigen3-dev`, or download from
  the [official website](https://eigen.tuxfamily.org/) and build from source
- OpenMP: Most time the compilers have already integrated with OpenMP. If your compiler does not support OpenMP,
  try `apt install libgomp-dev` or `apt install libomp-dev` for GCC or Clang separately.
- CUDA toolkit: It's recommended to install following
  the [official instructions](https://developer.nvidia.com/cuda-toolkit).
- CUTLASS: We have registered it to our git module, so you do not have to install by yourself.

## Compilation

Once you have You can use the following instruction to compile the project:

```bash
git clone git@github.com:openmlsys/openmlsys-cuda.git
cd openmlsys-cuda
git submodule init && git submodule sync
mkdir build && cd build
cmake ..
make -j4
```

## Examples

- `first_attempt`: The naive implementation
- `gemm`: Collection of implementations using different optimization tricks
- `fc_relu`: Example for fusing FC and ReLU by using CUTLASS
