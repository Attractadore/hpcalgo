#include "syclalgo.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

namespace {

TEST(Axpy, Saxpy) {
  size_t n = 1000;

  sycl::queue q{sycl::property::queue::in_order()};

  float alpha = 1.0f;

  std::vector<float> x(n);
  std::iota(x.begin(), x.end(), 1);

  std::vector<float> y = x;

  float *d_x = sycl::malloc_device<float>(n, q);
  q.copy(x.data(), d_x, n);

  float *d_y = sycl::malloc_device<float>(n, q);
  q.copy(y.data(), d_y, n);

  syclalgo::saxpy(q, n, alpha, d_x, d_y);

  std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                 [&](float x, float y) { return alpha * x + y; });

  std::vector<float> result(n);
  q.copy(d_y, result.data(), n).wait();

  sycl::free(d_x, q);
  sycl::free(d_y, q);

  EXPECT_EQ(y, result);
}

void test_exc_scan(sycl::queue &q, size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::exclusive_scan(data.begin(), data.end(), scan.begin(), 0);

  int *d_data = sycl::malloc_device<int>(n, q);
  sycl::event e = q.copy(data.data(), d_data, n);

  int *d_result = sycl::malloc_device<int>(n, q);

  e = syclalgo::exc_scan(q, n, d_data, d_result, {&e, 1});

  std::vector<int> result(n);
  q.copy(d_result, result.data(), n, e).wait();

  sycl::free(d_data, q);
  sycl::free(d_result, q);

  EXPECT_EQ(scan, result);
}

TEST(Scan, ExcScan) {
  sycl::queue q;
  {
    SCOPED_TRACE("exc_scan: single block");
    test_exc_scan(q, 100);
  }
  {
    SCOPED_TRACE("exc_scan: multi block");
    test_exc_scan(q, 1000);
  }
  {
    SCOPED_TRACE("exc_scan: multi level");
    test_exc_scan(q, 100'000);
  }
}

void test_inc_scan(sycl::queue &q, size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::inclusive_scan(data.begin(), data.end(), scan.begin());

  int *d_data = sycl::malloc_device<int>(n, q);
  sycl::event e = q.copy(data.data(), d_data, n);

  int *d_result = sycl::malloc_device<int>(n, q);

  e = syclalgo::inc_scan(q, n, d_data, d_result, {&e, 1});

  std::vector<int> result(n);
  q.copy(d_result, result.data(), n, e).wait();

  sycl::free(d_data, q);
  sycl::free(d_result, q);

  EXPECT_EQ(scan, result);
}

TEST(Scan, IncScan) {
  sycl::queue q;
  {
    SCOPED_TRACE("inc_scan: single block");
    test_inc_scan(q, 100);
  }
  {
    SCOPED_TRACE("inc_scan: multi block");
    test_inc_scan(q, 1000);
  }
  {
    SCOPED_TRACE("inc_scan: multi level");
    test_inc_scan(q, 100'000);
  }
}

} // namespace
