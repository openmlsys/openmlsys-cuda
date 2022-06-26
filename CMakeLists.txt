cmake_minimum_required(VERSION 3.12.4)
project(openmlsys-cuda CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
message("-- nvcc generates code for arch ${CUDA_ARCH_LIST}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${gflags_INCLUDE_DIR})

find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xcompiler -fopenmp")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif ()

add_executable(gemm gemm.cu gemm_final.cu gemm_hide_smem_latency.cu gemm_transpose_smem.cu gemm_use_smem.cu gemm_use_tile.cu gemm_use_128.cu gemm_naive.cu)
target_link_libraries(gemm ${CUDA_cublas_LIBRARY} OpenMP::OpenMP_CXX ${gflags_LIBRARIES})

add_executable(first_attempt first_attempt.cu)
target_link_libraries(first_attempt OpenMP::OpenMP_CXX)

set(CUTLASS_INCLUDE_DIR ./third_party/cutlass/include)
add_executable(fc_relu fc_relu.cu)
target_link_libraries(fc_relu PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(fc_relu PRIVATE ${CUTLASS_INCLUDE_DIR})
