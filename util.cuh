#ifndef GEMM_UTIL_CUH
#define GEMM_UTIL_CUH
namespace openmlsys {
template <int _m, int _n, int _k = 1>
struct Layout {
  static constexpr int m = _m;
  static constexpr int n = _n;
  static constexpr int k = _k;
};

struct __device_builtin__ __builtin_align__(16) float4 {
  float data[4];

  __host__ __device__ float operator[](unsigned idx) const { return data[idx]; }

  __host__ __device__ float &operator[](unsigned idx) { return data[idx]; }

  __host__ __device__ float4 operator*(float other) const {
    return float4{data[0] * other, data[1] * other, data[2] * other,
                  data[3] * other};
  }

  __host__ __device__ float4 operator+(const float4 &other) const {
    return float4{data[0] + other.data[0], data[1] + other.data[1],
                  data[2] + other.data[2], data[3] + other.data[3]};
  }
};

template <typename T>
struct __device_builtin__ Tensor2D {
  T *const __restrict__ ptr;
  const unsigned rows, cols;
  unsigned _rowOffset{0}, _colOffset{0};

  template <typename t>
  __host__ __device__ Tensor2D(t &&ptr, unsigned rows, unsigned cols)
      : ptr{reinterpret_cast<T *>(ptr)}, rows{rows}, cols{cols} {};

  template <typename t = T>
  __host__ __device__ void addOffset(unsigned rowOffset, unsigned colOffset) {
    _rowOffset += rowOffset;
    _colOffset += colOffset * sizeof(t) / sizeof(T);
  }

  __host__ __device__ bool validRowOffset(unsigned rowOffset) const {
    return (_rowOffset + rowOffset) < rows;
  }

  __host__ __device__ bool validColOffset(unsigned colOffset) const {
    return (_colOffset + colOffset) < cols;
  }

  __host__ __device__ bool validOffset(unsigned rowOffset,
                                       unsigned colOffset) const {
    return validRowOffset(rowOffset) && validColOffset(colOffset);
  }

  __host__ __device__ T &operator()(unsigned row, unsigned col) const {
    return ptr[_colOffset + col + (row + _rowOffset) * cols];
  }
};
}  // namespace openmlsys
#endif  // GEMM_UTIL_CUH
