#include "hipalgo.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <numeric>
#include <vector>

namespace {

void std_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> result(n);
  for (auto _ : state) {
    std::memcpy(result.data(), data.data(), sizeof(int) * n);
    auto out = result.data();
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void std_scan(benchmark::State &state) {
  size_t n = state.range(0);

  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> result(n);
  for (auto _ : state) {
    std::exclusive_scan(data.begin(), data.end(), result.begin(), 0);
    auto out = result.data();
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
  }
}

void hip_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  int *d_data;
  {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    HIP_TRY(hipMalloc(&d_data, sizeof(int) * n));
    HIP_TRY(
        hipMemcpy(d_data, data.data(), sizeof(int) * n, hipMemcpyHostToDevice));
  };

  int *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(int) * n));

  for (auto _ : state) {
    HIP_TRY(
        hipMemcpy(d_result, d_data, sizeof(int) * n, hipMemcpyDeviceToDevice));
    HIP_TRY(hipDeviceSynchronize());
  }

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));
}

void hipcub_scan(benchmark::State &state) {
  size_t n = state.range(0);

  int *d_data;
  {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    HIP_TRY(hipMalloc(&d_data, sizeof(int) * n));
    HIP_TRY(
        hipMemcpy(d_data, data.data(), sizeof(int) * n, hipMemcpyHostToDevice));
  };

  int *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(int) * n));

  int *d_scratch = nullptr;
  size_t scratch_size = 0;
  HIP_TRY(hipcub::DeviceScan::ExclusiveSum(d_scratch, scratch_size, &d_data[0],
                                           d_result, n));
  HIP_TRY(hipMalloc(&d_scratch, scratch_size));

  for (auto _ : state) {
    HIP_TRY(hipcub::DeviceScan::ExclusiveSum(d_scratch, scratch_size,
                                             &d_data[0], d_result, n));
    HIP_TRY(hipDeviceSynchronize());
  }

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));
  HIP_TRY(hipFree(d_scratch));
}

void recursive_scan(benchmark::State &state) {
  size_t n = state.range(0);

  int *d_data;
  {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    HIP_TRY(hipMalloc(&d_data, sizeof(int) * n));
    HIP_TRY(
        hipMemcpy(d_data, data.data(), sizeof(int) * n, hipMemcpyHostToDevice));
  };

  int *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(int) * n));

  for (auto _ : state) {
    hipalgo::exclusive_recursive_scan(n, d_data, d_result);
    HIP_TRY(hipDeviceSynchronize());
  }

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));
}

constexpr size_t MB = 1024 * 1024;

constexpr size_t MIN_COUNT = 1 * MB / sizeof(int);
constexpr size_t MAX_COUNT = 512 * MB / sizeof(int);

BENCHMARK(std_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(std_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(hip_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(hipcub_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(recursive_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);

} // namespace
