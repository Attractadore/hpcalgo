#include "syclalgo.hpp"

namespace syclalgo {

namespace {

template <typename P, typename T> constexpr bool PointerOrBufferRefImpl = false;

template <typename T> constexpr bool PointerOrBufferRefImpl<T *&, T> = true;

template <typename T>
constexpr bool PointerOrBufferRefImpl<sycl::buffer<T> &, T> = true;

template <typename T>
constexpr bool PointerOrBufferRefImpl<sycl::buffer<T> &, const T> = true;

template <typename P, typename T>
concept PointerOrBufferRef = PointerOrBufferRefImpl<P, T>;

template <typename P, typename T> constexpr bool PointerOrAccessorImpl = false;

template <typename T> constexpr bool PointerOrAccessorImpl<T, T *> = true;

template <typename T>
constexpr bool PointerOrAccessorImpl<
    sycl::accessor<T, 1, sycl::access_mode::read>, const T> = true;

template <typename T>
constexpr bool
    PointerOrAccessorImpl<sycl::accessor<T, 1, sycl::access_mode::write>, T> =
        true;

template <typename T>
constexpr bool PointerOrAccessorImpl<
    sycl::accessor<T, 1, sycl::access_mode::read_write>, T> = true;

template <typename P, typename T>
concept PointerOrAccessor = PointerOrAccessorImpl<P, T>;

template <typename T>
auto access(sycl::buffer<T> &buffer, sycl::handler &cg,
            auto tag = sycl::read_write,
            const sycl::property_list &props = {}) {
  return sycl::accessor(buffer, cg, tag, props);
}

template <typename T>
auto access(T *ptr, sycl::handler &, auto = sycl::read_write,
            const sycl::property_list & = {}) {
  return ptr;
}

static thread_local std::vector<sycl::event> kernel_dependences;

template <typename T>
auto axpy(sycl::queue &q, size_t n, T alpha,
          PointerOrBufferRef<const T> auto &&x, PointerOrBufferRef<T> auto &&y,
          std::span<const sycl::event> dependences = {}) -> sycl::event {
  if (n == 0) {
    return {};
  }

  return q.submit([&](sycl::handler &cg) {
    auto ax = access(x, cg, sycl::read_only);
    auto ay = access(y, cg, sycl::read_write);
    kernel_dependences.assign(dependences.begin(), dependences.end());
    cg.depends_on(kernel_dependences);
    kernel_dependences.clear();
    cg.parallel_for(sycl::range(n), [=](sycl::id<1> idx) {
      ay[idx] = alpha * ax[idx] + ay[idx];
    });
  });
}

} // namespace

void saxpy(sycl::queue &q, size_t n, float alpha, sycl::buffer<float> &x,
           sycl::buffer<float> &y) {
  axpy(q, n, alpha, x, y);
}

auto saxpy(sycl::queue &q, size_t n, float alpha, const float *d_x, float *d_y,
           std::span<const sycl::event> dependences) -> sycl::event {
  return axpy(q, n, alpha, d_x, d_y, dependences);
}

} // namespace syclalgo
