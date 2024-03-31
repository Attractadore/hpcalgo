#include "syclalgo.hpp"
#include <benchmark/benchmark.h>
#if ONEMKL
#include <oneapi/mkl.hpp>
#endif

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

void sycl_memcpy(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q({sycl::property::queue::in_order()});

  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_x, n);
  };

  for (auto _ : state) {
    q.memcpy(d_y, d_x, sizeof(float) * n).wait();
  }

  sycl::free(d_x, q);
  sycl::free(d_y, q);
}

#if ONEMKL
void onemkl_saxpy(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q({sycl::property::queue::in_order()});

  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_x, n);
    q.copy(data.data(), d_y, n);
  };

  for (auto _ : state) {
    float alpha = 1.0f;
    oneapi::mkl::blas::row_major::axpy(q, n, alpha, d_x, 1, d_y, 1).wait();
  }

  sycl::free(d_x, q);
  sycl::free(d_y, q);
}
#endif

void saxpy(benchmark::State &state) {
  size_t n = state.range(0);

  sycl::queue q({sycl::property::queue::in_order()});

  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);
  {
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1);
    q.copy(data.data(), d_x, n);
    q.copy(data.data(), d_y, n);
  };

  for (auto _ : state) {
    float alpha = 1.0f;
    syclalgo::saxpy(q, {}, n, alpha, d_x, d_y).wait();
  }

  sycl::free(d_x, q);
  sycl::free(d_y, q);
}

constexpr size_t MB = 1024 * 1024;

constexpr size_t MIN_COUNT = 1 * MB / sizeof(float);
constexpr size_t MAX_COUNT = 512 * MB / sizeof(int);

BENCHMARK(std_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(std_tranform)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
BENCHMARK(sycl_memcpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
#if ONEMKL
BENCHMARK(onemkl_saxpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);
#endif
BENCHMARK(saxpy)->RangeMultiplier(2)->Range(MIN_COUNT, MAX_COUNT);

} // namespace
