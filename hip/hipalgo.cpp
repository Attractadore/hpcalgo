#include "hipalgo.hpp"
#include <hip/hip_runtime.h>

namespace hipalgo {

namespace {

constexpr size_t ceil_div(size_t num, size_t denom) {
  return num / denom + (num % denom != 0);
}

template <typename T> constexpr void swap(T &lhs, T &rhs) {
  auto tmp = std::move(lhs);
  lhs = std::move(rhs);
  rhs = std::move(tmp);
}

} // namespace

namespace {

template <typename T>
__global__ void kernel_axpy(int n, T a, const T *x, T *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

} // namespace

void saxpy(size_t n, float a, const float *d_x, float *d_y) {
  constexpr size_t BLOCK_SIZE = 128;

  size_t num_blocks = ceil_div(n, BLOCK_SIZE);
  if (num_blocks == 0) {
    return;
  }

  kernel_axpy<<<num_blocks, BLOCK_SIZE>>>(n, a, d_x, d_y);
  HIP_TRY(hipGetLastError());
}

namespace {

enum class ScanType {
  Exclusive,
  Inclusive,
};

template <ScanType ST, int BT, int TE>
__global__ void kernel_scan(int n, const int *data, int *out, int *block_sum) {
  constexpr int BE = BT * TE;
  __shared__ int shm[BE * 2];
  int *shm_src = shm;
  int *shm_dst = shm + BE;

  for (int i = 0; i < TE; ++i) {
    int lidx = i * BT + threadIdx.x;
    if constexpr (ST == ScanType::Exclusive) {
      int gidx = blockIdx.x * BE + lidx - 1;
      shm_src[lidx] = gidx >= 0 && gidx < n ? data[gidx] : 0;
    } else if constexpr (ST == ScanType::Inclusive) {
      int gidx = blockIdx.x * BE + lidx;
      shm_src[lidx] = gidx < n ? data[gidx] : 0;
    }
  }
  __syncthreads();

  for (int stride = 1; stride < BE; stride *= 2) {
    for (int i = 0; i < TE; ++i) {
      int dst = BE - (i * BT + threadIdx.x) - 1;
      int value = shm_src[dst];
      int src = dst - stride;
      if (src >= 0) {
        value += shm_src[src];
      }
      shm_dst[dst] = value;
    }
    __syncthreads();
    swap(shm_src, shm_dst);
  }

  for (int i = 0; i < TE; ++i) {
    int lidx = i * BT + threadIdx.x;
    int gidx = blockIdx.x * BE + lidx;
    if (gidx < n) {
      out[gidx] = shm_src[lidx];
    }
  }

  if (block_sum && threadIdx.x == 0) {
    block_sum[blockIdx.x] = shm_src[BE - 1];
  }
}

template <int BT, int TE>
__global__ void kernel_scan_add_block_sum(int n, int *data,
                                          const int *block_sum) {

  constexpr int BE = BT * TE;
  for (int i = 0; i < TE; ++i) {
    int gidx = BE * blockIdx.x + i * BT + threadIdx.x;
    if (gidx < n) {
      data[gidx] += block_sum[blockIdx.x];
    }
  }
}

template <ScanType ST> void recursive_scan(int n, const int *data, int *out) {
  constexpr size_t BLOCK_SIZE = 64;
  constexpr size_t THREAD_ELEMENTS = 8;
  constexpr size_t BLOCK_ELEMENTS = BLOCK_SIZE * THREAD_ELEMENTS;

  constexpr auto kernel = kernel_scan<ST, BLOCK_SIZE, THREAD_ELEMENTS>;

  size_t num_blocks = ceil_div(n, BLOCK_ELEMENTS);
  if (num_blocks == 0) {
    return;
  }

  if (num_blocks == 1) {
    kernel<<<1, BLOCK_SIZE>>>(n, data, out, nullptr);
    HIP_TRY(hipGetLastError());
    return;
  }

  int *block_sum;
  HIP_TRY(hipMalloc(&block_sum, sizeof(int) * num_blocks));

  kernel<<<num_blocks, BLOCK_SIZE>>>(n, data, out, block_sum);
  HIP_TRY(hipGetLastError());

  recursive_scan<ScanType::Exclusive>(num_blocks, block_sum, block_sum);

  kernel_scan_add_block_sum<BLOCK_SIZE, THREAD_ELEMENTS>
      <<<num_blocks, BLOCK_SIZE>>>(n, out, block_sum);
  HIP_TRY(hipGetLastError());

  HIP_TRY(hipFree(block_sum));
}

} // namespace
  //
void exclusive_scan(size_t n, const int *d_data, int *d_out) {
  exclusive_recursive_scan(n, d_data, d_out);
}

void inclusive_scan(size_t n, const int *d_data, int *d_out) {
  inclusive_recursive_scan(n, d_data, d_out);
}

void exclusive_recursive_scan(size_t n, const int *d_data, int *d_out) {
  recursive_scan<ScanType::Exclusive>(n, d_data, d_out);
}

void inclusive_recursive_scan(size_t n, const int *d_data, int *d_out) {
  recursive_scan<ScanType::Inclusive>(n, d_data, d_out);
}

} // namespace hipalgo
