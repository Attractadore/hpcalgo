#include "syclalgo.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

namespace {

TEST(Axpy, Saxpy) {
  size_t n = 1000;

  sycl::queue q;

  float alpha = 1.0f;

  std::vector<float> x(n);
  std::iota(x.begin(), x.end(), 1);

  std::vector<float> y = x;

  sycl::buffer<float> bx(x.cbegin(), x.cend());

  sycl::buffer<float> by(y.cbegin(), y.cend());

  syclalgo::saxpy(q, n, alpha, bx, by);

  std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                 [&](float x, float y) { return alpha * x + y; });

  std::vector<float> result(n);
  q.copy(sycl::accessor(by), result.data()).wait();

  EXPECT_EQ(y, result);
}

TEST(Axpy, SaxpyUSM) {
  size_t n = 1000;

  sycl::queue q{sycl::property::queue::in_order()};
  if (not q.get_device().has(sycl::aspect::usm_device_allocations)) {
    GTEST_SKIP() << "USM unsupported";
  }

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

void test_exc_scan(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::exclusive_scan(data.begin(), data.end(), scan.begin(), 0);

  sycl::queue q;

  sycl::buffer<int> bdata(data.begin(), data.end());
  sycl::buffer<int> bresult(n);

  syclalgo::exc_scan(q, n, bdata, bresult);

  std::vector<int> result(n);
  q.copy(sycl::accessor(bresult), result.data()).wait();

  EXPECT_EQ(scan, result);
}

void test_exc_scan_usm(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::exclusive_scan(data.begin(), data.end(), scan.begin(), 0);

  sycl::queue q{sycl::property::queue::in_order()};

  int *d_data = sycl::malloc_device<int>(n, q);
  q.copy(data.data(), d_data, n);

  int *d_result = sycl::malloc_device<int>(n, q);

  syclalgo::exc_scan(q, n, d_data, d_result);

  std::vector<int> result(n);
  q.copy(d_result, result.data(), n).wait();

  sycl::free(d_data, q);
  sycl::free(d_result, q);

  EXPECT_EQ(scan, result);
}

TEST(Scan, ExcScan) {
  {
    SCOPED_TRACE("exc_scan: single block");
    test_exc_scan(100);
  }
  {
    SCOPED_TRACE("exc_scan: multi block");
    test_exc_scan(1000);
  }
  {
    SCOPED_TRACE("exc_scan: multi level");
    test_exc_scan(100'000);
  }
}

TEST(Scan, ExcScanUSM) {
  {
    SCOPED_TRACE("exc_scan_usm: single block");
    test_exc_scan_usm(100);
  }
  {
    SCOPED_TRACE("exc_scan_usm: multi block");
    test_exc_scan_usm(1000);
  }
  {
    SCOPED_TRACE("exc_scan_usm: multi level");
    test_exc_scan_usm(100'000);
  }
}

void test_inc_scan(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::inclusive_scan(data.begin(), data.end(), scan.begin());

  sycl::queue q;

  sycl::buffer<int> bdata(data.begin(), data.end());
  sycl::buffer<int> bresult(n);

  syclalgo::inc_scan(q, n, bdata, bresult);

  std::vector<int> result(n);
  q.copy(sycl::accessor(bresult), result.data()).wait();

  EXPECT_EQ(scan, result);
}

void test_inc_scan_usm(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::inclusive_scan(data.begin(), data.end(), scan.begin());

  sycl::queue q{sycl::property::queue::in_order()};

  int *d_data = sycl::malloc_device<int>(n, q);
  q.copy(data.data(), d_data, n);

  int *d_result = sycl::malloc_device<int>(n, q);

  syclalgo::inc_scan(q, n, d_data, d_result);

  std::vector<int> result(n);
  q.copy(d_result, result.data(), n).wait();

  sycl::free(d_data, q);
  sycl::free(d_result, q);

  EXPECT_EQ(scan, result);
}

TEST(Scan, IncScan) {
  {
    SCOPED_TRACE("inc_scan: single block");
    test_inc_scan(100);
  }
  {
    SCOPED_TRACE("inc_scan: multi block");
    test_inc_scan(1000);
  }
  {
    SCOPED_TRACE("inc_scan: multi level");
    test_inc_scan(100'000);
  }
}

TEST(Scan, IncScanUSM) {
  {
    SCOPED_TRACE("inc_scan_usm: single block");
    test_inc_scan_usm(100);
  }
  {
    SCOPED_TRACE("inc_scan_usm: multi block");
    test_inc_scan_usm(1000);
  }
  {
    SCOPED_TRACE("inc_scan_usm: multi level");
    test_inc_scan_usm(100'000);
  }
}

} // namespace
