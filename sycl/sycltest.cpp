#include "syclalgo.hpp"
#include <gtest/gtest.h>

namespace {

TEST(Axpy, Saxpy) {
  size_t n = 1000;

  sycl::queue q({sycl::property::queue::in_order()});

  float alpha = 1.0f;

  std::vector<float> x(n);
  std::iota(x.begin(), x.end(), 1);

  std::vector<float> y = x;

  float *d_x = sycl::malloc_device<float>(n, q);
  q.copy(x.data(), d_x, n);

  float *d_y = sycl::malloc_device<float>(n, q);
  q.copy(y.data(), d_y, n);

  syclalgo::saxpy(q, {}, n, alpha, d_x, d_y);

  std::vector<float> result(n);
  q.copy(d_y, result.data(), n).wait();

  sycl::free(d_x, q);
  sycl::free(d_y, q);

  std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                 [&](float x, float y) { return alpha * x + y; });

  EXPECT_EQ(y, result);
}

} // namespace
