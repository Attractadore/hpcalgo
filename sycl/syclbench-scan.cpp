#include "syclalgo.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <numeric>
#if ONEDPL
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

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

void sycl_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q;

  sycl::buffer<float> bx(n);
  sycl::buffer<float> by(n);
  {
    sycl::host_accessor ax(bx, sycl::write_only, sycl::no_init);
    std::iota(ax.begin(), ax.end(), 1);
  };

  sycl::accessor ax(bx, sycl::read_only);
  sycl::accessor ay(by, sycl::write_only);
  for (auto _ : state) {
    q.copy(ax, ay).wait();
  }
}

void sycl_memcpy_usm(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q{sycl::property::queue::in_order()};
  if (not q.get_device().has(sycl::aspect::usm_device_allocations)) {
    state.SkipWithError("USM unsupported");
    return;
  }

  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_x, n);
  };

  for (auto _ : state) {
    q.copy(d_x, d_y, n).wait();
  }

  sycl::free(d_x, q);
  sycl::free(d_y, q);
}

#if ONEDPL

void onedpl_scan(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q;

  sycl::buffer<int> bdata(n);
  {
    sycl::host_accessor adata(bdata, sycl::write_only, sycl::no_init);
    std::iota(adata.begin(), adata.end(), 1);
  }

  sycl::buffer<int> bresult(n);

  auto policy = oneapi::dpl::execution::make_device_policy(q);
  for (auto _ : state) {
    oneapi::dpl::experimental::exclusive_scan_async(
        policy, oneapi::dpl::begin(bdata), oneapi::dpl::end(bdata),
        oneapi::dpl::begin(bresult), 0)
        .wait();
  }
}

void onedpl_scan_usm(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q{sycl::property::queue::in_order()};

  int *d_data = sycl::malloc_device<int>(n, q);
  {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_data, n);
  };

  int *d_result = sycl::malloc_device<int>(n, q);

  auto policy = oneapi::dpl::execution::make_device_policy(q);
  for (auto _ : state) {
    oneapi::dpl::experimental::exclusive_scan_async(policy, d_data, d_data + n,
                                                    d_result, 0)
        .wait();
  }

  sycl::free(d_data, q);
  sycl::free(d_result, q);
}

#endif

void exc_scan(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q;

  sycl::buffer<int> bdata(n);
  {
    sycl::host_accessor adata(bdata, sycl::write_only, sycl::no_init);
    std::iota(adata.begin(), adata.end(), 1);
  }

  sycl::buffer<int> bresult(n);

  for (auto _ : state) {
    syclalgo::exc_scan(q, n, bdata, bresult);
    q.wait();
  }
}

void exc_scan_usm(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q{sycl::property::queue::in_order()};

  int *d_data = sycl::malloc_device<int>(n, q);
  {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_data, n);
  };

  int *d_result = sycl::malloc_device<int>(n, q);

  for (auto _ : state) {
    syclalgo::exc_scan(q, n, d_data, d_result).wait();
  }

  sycl::free(d_data, q);
  sycl::free(d_result, q);
}

constexpr size_t MB = 1024 * 1024;

constexpr size_t MIN_COUNT = 1 * MB / sizeof(int);
constexpr size_t MAX_COUNT = 512 * MB / sizeof(int);

BENCHMARK(std_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(std_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(sycl_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(sycl_memcpy_usm)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
#if ONEDPL
BENCHMARK(onedpl_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(onedpl_scan_usm)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
#endif
BENCHMARK(exc_scan)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(exc_scan_usm)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);

} // namespace
