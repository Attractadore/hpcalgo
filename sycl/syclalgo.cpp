#include "syclalgo.hpp"
#include <thread>
#include <utility>

namespace syclalgo {

namespace {

static thread_local std::vector<sycl::event> kernel_dependences;

constexpr auto ceil_div(size_t num, size_t denom) -> size_t {
  return num / denom + (num % denom != 0);
}

} // namespace

namespace {

template <typename T>
auto axpy(sycl::queue &q, size_t n, T alpha, const T *d_x, T *d_y,
          std::span<const sycl::event> dependences = {}) -> sycl::event {
  if (n == 0) {
    return {};
  }

  return q.submit([&](sycl::handler &cg) {
    kernel_dependences.assign(dependences.begin(), dependences.end());
    cg.depends_on(kernel_dependences);
    kernel_dependences.clear();
    cg.parallel_for(
        n, [=](sycl::id<1> idx) { d_y[idx] = alpha * d_x[idx] + d_y[idx]; });
  });
}

} // namespace

auto saxpy(sycl::queue &q, size_t n, float alpha, const float *d_x, float *d_y,
           std::span<const sycl::event> dependences) -> sycl::event {
  return axpy(q, n, alpha, d_x, d_y, dependences);
}

namespace {

enum class ScanType {
  Exclusive,
  Inclusive,
};

constexpr int SCAN_WG_SIZE = 64;
constexpr int SCAN_WI_ELEMS = 8;
constexpr int SCAN_WG_ELEMS = SCAN_WG_SIZE * SCAN_WI_ELEMS;

template <ScanType ST>
auto scan_impl(sycl::queue &q, int n, const int *d_data, int *d_out,
               int *d_scratch, std::span<const sycl::event> dependences = {})
    -> sycl::event {
  int num_groups = ceil_div(n, SCAN_WG_ELEMS);
  sycl::nd_range<1> range = {num_groups * SCAN_WG_SIZE, SCAN_WG_SIZE};

  int *d_block_sum = num_groups > 1 ? d_scratch : nullptr;

  sycl::event e = q.submit([&](sycl::handler &cg) {
    sycl::local_accessor<int> shm(SCAN_WG_ELEMS * 2, cg);

    kernel_dependences.assign(dependences.begin(), dependences.end());
    cg.depends_on(kernel_dependences);
    kernel_dependences.clear();

    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      sycl::local_ptr<int> shm_src(shm);
      sycl::local_ptr<int> shm_dst = shm_src + SCAN_WG_ELEMS;

      auto g = id.get_group();
      int lid = id.get_local_id();
      int gid = g.get_group_id();

      for (int i = 0; i < SCAN_WI_ELEMS; ++i) {
        int lidx = i * SCAN_WG_SIZE + lid;
        int gidx = gid * SCAN_WG_ELEMS + lidx;
        if constexpr (ST == ScanType::Exclusive) {
          gidx--;
          shm_src[lidx] = gidx >= 0 && gidx < n ? d_data[gidx] : 0;
        } else if constexpr (ST == ScanType::Inclusive) {
          shm_src[lidx] = gidx < n ? d_data[gidx] : 0;
        }
      }
      sycl::group_barrier(g);

      for (int stride = 1; stride < SCAN_WG_ELEMS; stride *= 2) {
        for (int i = 0; i < SCAN_WI_ELEMS; ++i) {
          int dst = SCAN_WG_ELEMS - (i * SCAN_WG_SIZE + lid) - 1;
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

      for (int i = 0; i < SCAN_WI_ELEMS; ++i) {
        int lidx = i * SCAN_WG_SIZE + lid;
        int gidx = gid * SCAN_WG_ELEMS + lidx;
        if (gidx < n) {
          d_out[gidx] = shm_src[lidx];
        }
      }

      if (d_block_sum && g.leader()) {
        d_block_sum[gid] = shm_src[SCAN_WG_ELEMS - 1];
      }
    });
  });

  if (!d_block_sum) {
    return e;
  }

  e = scan_impl<ScanType::Exclusive>(q, num_groups, d_block_sum, d_block_sum,
                                     d_scratch + num_groups, {&e, 1});

  return q.submit([&](sycl::handler &cg) {
    cg.depends_on(e);
    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      auto g = id.get_group();
      int lid = id.get_local_id();
      int gid = g.get_group_id();
      for (int i = 0; i < SCAN_WI_ELEMS; ++i) {
        int lidx = i * SCAN_WG_SIZE + lid;
        int gidx = gid * SCAN_WG_ELEMS + lidx;
        if (gidx < n) {
          d_out[gidx] += d_block_sum[gid];
        }
      }
    });
  });
}

template <ScanType ST>
auto scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
          std::span<const sycl::event> dependences = {}) -> sycl::event {
  if (n == 0) {
    return {};
  }

  size_t scratch_cnt = 0;
  size_t num_groups = n;
  do {
    num_groups = ceil_div(num_groups, SCAN_WG_ELEMS);
    scratch_cnt += num_groups;
  } while (num_groups > 1);

  int *d_scratch =
      scratch_cnt > 1 ? sycl::malloc_device<int>(scratch_cnt, q) : nullptr;

  sycl::event e = scan_impl<ST>(q, n, d_data, d_out, d_scratch, dependences);

  std::thread([q, e, d_scratch]() mutable {
    e.wait();
    sycl::free(d_scratch, q);
  }).detach();

  return e;
}

} // namespace

auto exc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences) -> sycl::event {
  return scan<ScanType::Exclusive>(q, n, d_data, d_out, dependences);
}

auto inc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences) -> sycl::event {
  return scan<ScanType::Inclusive>(q, n, d_data, d_out, dependences);
}

} // namespace syclalgo
