#include "syclalgo.hpp"

namespace syclalgo {

namespace {

static thread_local std::vector<sycl::event> kernel_dependences;

template <typename T>
auto axpy(sycl::queue &q, std::span<const sycl::event> dependences, size_t n,
          T a, const T *d_x, T *d_y) -> sycl::event {
  if (n == 0) {
    return {};
  }

  kernel_dependences.assign(dependences.begin(), dependences.end());
  sycl::event e = q.parallel_for(n, kernel_dependences, [=](sycl::id<1> idx) {
    d_y[idx] = a * d_x[idx] + d_y[idx];
  });
  kernel_dependences.clear();

  return e;
}

} // namespace

auto saxpy(sycl::queue &q, std::span<const sycl::event> dependences, size_t n,
           float a, const float *d_x, float *d_y) -> sycl::event {
  return axpy(q, dependences, n, a, d_x, d_y);
}

} // namespace syclalgo
