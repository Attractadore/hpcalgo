#include "hipalgo.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <numeric>
#include <vector>

namespace {

void std_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  std::vector<float> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<float> result(n);
  for (auto _ : state) {
    std::memcpy(result.data(), data.data(), sizeof(float) * n);
    auto out = result.data();
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void std_tranform(benchmark::State &state) {
  size_t n = state.range(0);

  std::vector<float> x(n);
  std::iota(x.begin(), x.end(), 1);

  std::vector<float> y(n);
  std::iota(y.begin(), y.end(), 1);

  for (auto _ : state) {
    float a = 1.0f;
    std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                   [&](float x, float y) { return a * x + y; });
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
}

void hip_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  float *d_data;
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);
    HIP_TRY(hipMalloc(&d_data, sizeof(float) * n));
    HIP_TRY(hipMemcpy(d_data, data.data(), sizeof(float) * n,
                      hipMemcpyHostToDevice));
  };

  float *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(float) * n));

  for (auto _ : state) {
    HIP_TRY(hipMemcpy(d_result, d_data, sizeof(float) * n,
                      hipMemcpyDeviceToDevice));
    HIP_TRY(hipDeviceSynchronize());
  }

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));
}

#define HIPBLAS_TRY(...)                                                       \
  do {                                                                         \
    hipblasStatus_t res = __VA_ARGS__;                                         \
    if (res != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "%s:%d, %s failed: %d (%s)\n", __FILE__, __LINE__,  \
                   #__VA_ARGS__, res, hipblasStatusToString(res));             \
      std::exit(-1);                                                           \
    }                                                                          \
  } while (0)

void hipblas_saxpy(benchmark::State &state) {
  size_t n = state.range(0);

  float *d_x;
  float *d_y;
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);

    HIP_TRY(hipMalloc(&d_x, sizeof(float) * n));
    HIP_TRY(
        hipMemcpy(d_x, data.data(), sizeof(float) * n, hipMemcpyHostToDevice));

    HIP_TRY(hipMalloc(&d_y, sizeof(float) * n));
    HIP_TRY(
        hipMemcpy(d_y, data.data(), sizeof(float) * n, hipMemcpyHostToDevice));
  };

  hipblasHandle_t blas;
  HIPBLAS_TRY(hipblasCreate(&blas));

  for (auto _ : state) {
    float alpha = 1.0f;
    HIPBLAS_TRY(hipblasSaxpy(blas, n, &alpha, d_x, 1, d_y, 1));
    HIP_TRY(hipDeviceSynchronize());
  }

  HIPBLAS_TRY(hipblasDestroy(blas));

  HIP_TRY(hipFree(d_x));
  HIP_TRY(hipFree(d_y));
}

void saxpy(benchmark::State &state) {
  size_t n = state.range(0);

  float *d_x;
  float *d_y;
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);

    HIP_TRY(hipMalloc(&d_x, sizeof(float) * n));
    HIP_TRY(
        hipMemcpy(d_x, data.data(), sizeof(float) * n, hipMemcpyHostToDevice));

    HIP_TRY(hipMalloc(&d_y, sizeof(float) * n));
    HIP_TRY(
        hipMemcpy(d_y, data.data(), sizeof(float) * n, hipMemcpyHostToDevice));
  };

  for (auto _ : state) {
    float alpha = 1.0f;
    hipalgo::saxpy(n, alpha, d_x, d_y);
    HIP_TRY(hipDeviceSynchronize());
  }

  HIP_TRY(hipFree(d_x));
  HIP_TRY(hipFree(d_y));
}

constexpr size_t MB = 1024 * 1024;

constexpr size_t MIN_COUNT = 1 * MB / sizeof(int);
constexpr size_t MAX_COUNT = 512 * MB / sizeof(int);

BENCHMARK(std_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(std_tranform)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(hip_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(hipblas_saxpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(saxpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);

} // namespace
