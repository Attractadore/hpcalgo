#include "syclalgo.hpp"
#include <optional>
#include <utility>

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

constexpr auto ceil_div(size_t num, size_t denom) -> size_t {
  return num / denom + (num % denom != 0);
}

} // namespace

namespace {

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
    cg.parallel_for(
        n, [=](sycl::id<1> idx) { ay[idx] = alpha * ax[idx] + ay[idx]; });
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

namespace {

enum class ScanType {
  Exclusive,
  Inclusive,
};

template <ScanType ST>
auto scan(sycl::queue &q, size_t n, PointerOrBufferRef<const int> auto &&data,
          PointerOrBufferRef<int> auto &&out,
          std::span<const sycl::event> dependences = {}) -> sycl::event {
  constexpr size_t WG_SIZE = 128;
  constexpr size_t WI_ELEMS = 4;
  constexpr size_t WG_ELEMS = WG_SIZE * WI_ELEMS;

  size_t num_groups = ceil_div(n, WG_ELEMS);
  if (num_groups == 0) {
    return {};
  }
  sycl::nd_range<1> range = {num_groups * WG_SIZE, WG_SIZE};

  std::optional<sycl::buffer<int>> block_sum;
  if (num_groups > 1) {
    block_sum = sycl::buffer<int>(num_groups);
  }

  sycl::event e = q.submit([&](sycl::handler &cg) {
    auto data_acc = access(data, cg, sycl::read_only);
    auto out_acc = access(out, cg, sycl::write_only);

    sycl::accessor<int, 1, sycl::access_mode::write> block_sum_acc;
    if (block_sum) {
      block_sum_acc =
          sycl::accessor(*block_sum, cg, sycl::write_only, sycl::no_init);
    }

    sycl::local_accessor<int> shm(WG_ELEMS * 2, cg);

    kernel_dependences.assign(dependences.begin(), dependences.end());
    cg.depends_on(kernel_dependences);
    kernel_dependences.clear();

    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      sycl::local_ptr<int> shm_src(shm);
      sycl::local_ptr<int> shm_dst = shm_src + WG_ELEMS;

      auto g = id.get_group();

      for (int i = 0; i < WI_ELEMS; ++i) {
        int lidx = i * WG_SIZE + id.get_local_id();
        int gidx = g.get_group_id() * WG_ELEMS + lidx;
        if constexpr (ST == ScanType::Exclusive) {
          gidx--;
          shm_src[lidx] = gidx >= 0 && gidx < n ? data_acc[gidx] : 0;
        } else if constexpr (ST == ScanType::Inclusive) {
          shm_src[lidx] = gidx < n ? data_acc[gidx] : 0;
        }
      }
      sycl::group_barrier(g);

      for (int stride = 1; stride < WG_ELEMS; stride *= 2) {
        for (int i = 0; i < WI_ELEMS; ++i) {
          int dst = WG_ELEMS - (i * WG_SIZE + id.get_local_id()) - 1;
          int value = shm_src[dst];
          int src = dst - stride;
          if (src >= 0) {
            value += shm_src[src];
          }
          shm_dst[dst] = value;
        }
        sycl::group_barrier(g);
        std::swap(shm_src, shm_dst);
      }

      for (int i = 0; i < WI_ELEMS; ++i) {
        int lidx = i * WG_SIZE + id.get_local_id();
        int gidx = g.get_group_id() * WG_ELEMS + lidx;
        if (gidx < n) {
          out_acc[gidx] = shm_src[lidx];
        }
      }

      if (not block_sum_acc.empty() && g.leader()) {
        block_sum_acc[g.get_group_id()] = shm_src[WG_ELEMS - 1];
      }
    });
  });

  if (!block_sum) {
    return e;
  }

  scan<ScanType::Exclusive>(q, num_groups, *block_sum, *block_sum, {&e, 1});

  return q.submit([&](sycl::handler &cg) {
    sycl::accessor block_sum_acc(*block_sum, cg, sycl::read_only);
    auto out_acc = access(out, cg, sycl::read_write);

    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      auto g = id.get_group();
      for (int i = 0; i < WI_ELEMS; ++i) {
        int lidx = i * WG_SIZE + g.get_local_id();
        int gidx = g.get_group_id() * WG_ELEMS + lidx;
        if (gidx < n) {
          out_acc[gidx] += block_sum_acc[g.get_group_id()];
        }
      }
    });
  });
}

} // namespace

void exc_scan(sycl::queue &q, size_t n, sycl::buffer<int> &data,
              sycl::buffer<int> &out) {
  scan<ScanType::Exclusive>(q, n, data, out);
}

void inc_scan(sycl::queue &q, size_t n, sycl::buffer<int> &data,
              sycl::buffer<int> &out) {
  scan<ScanType::Inclusive>(q, n, data, out);
}

auto exc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences) -> sycl::event {
  return scan<ScanType::Exclusive>(q, n, d_data, d_out, dependences);
}

auto inc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences) -> sycl::event {
  return scan<ScanType::Inclusive>(q, n, d_data, d_out, dependences);
}

} // namespace syclalgo
