#include "syclalgo.hpp"
#include <thread>

namespace syclalgo {

namespace {

void depends_on(sycl::handler &cg, std::span<const sycl::event> dependences) {
  static thread_local std::vector<sycl::event> kernel_dependences;
  kernel_dependences.assign(dependences.begin(), dependences.end());
  cg.depends_on(kernel_dependences);
  kernel_dependences.clear();
}

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
    depends_on(cg, dependences);
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

template <int BLOCK_SIZE, int ELEMS>
void group_inclusive_scan(sycl::group<1> g, sycl::local_ptr<int> shm) {
  constexpr int BLOCK_ELEMS = BLOCK_SIZE * ELEMS;

  int lid = g.get_local_id();

  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    for (int i = 0; i < ELEMS; ++i) {
      int dst = BLOCK_ELEMS - (i * BLOCK_SIZE + lid) - 1;
      int src = dst - stride;
      int dst_value, src_value;
      if (src >= 0) {
        dst_value = shm[dst];
        src_value = shm[src];
      }
      sycl::group_barrier(g);
      if (src >= 0) {
        shm[dst] = src_value + dst_value;
      }
    }
    sycl::group_barrier(g);
  }

  for (int stride = BLOCK_SIZE; stride < BLOCK_ELEMS; stride *= 2) {
    for (int i = 0; i < ELEMS; ++i) {
      int dst = BLOCK_ELEMS - (i * BLOCK_SIZE + lid) - 1;
      int src = dst - stride;
      if (src >= 0) {
        shm[dst] += shm[src];
      }
    }
    sycl::group_barrier(g);
  }
}

template <ScanType ST, int BLOCK_SIZE, int ELEMS>
auto recursive_scan_impl(sycl::queue &q, int n, const int *d_data, int *d_out,
                         int *d_scratch,
                         std::span<const sycl::event> dependences = {})
    -> sycl::event {
  constexpr int BLOCK_ELEMS = BLOCK_SIZE * ELEMS;

  int num_groups = ceil_div(n, BLOCK_ELEMS);
  sycl::nd_range<1> range = {num_groups * BLOCK_SIZE, BLOCK_SIZE};

  int *d_block_sum = num_groups > 1 ? d_scratch : nullptr;

  sycl::event e = q.submit([&](sycl::handler &cg) {
    sycl::local_accessor<int> shm(BLOCK_ELEMS, cg);

    depends_on(cg, dependences);

    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      auto g = id.get_group();
      int lid = g.get_local_id();
      int bid = g.get_group_id();

      for (int i = 0; i < ELEMS; ++i) {
        int lidx = i * BLOCK_SIZE + lid;
        int gidx = bid * BLOCK_ELEMS + lidx;
        if constexpr (ST == ScanType::Exclusive) {
          gidx--;
          shm[lidx] = gidx >= 0 && gidx < n ? d_data[gidx] : 0;
        } else if constexpr (ST == ScanType::Inclusive) {
          shm[lidx] = gidx < n ? d_data[gidx] : 0;
        }
      }
      sycl::group_barrier(g);

      group_inclusive_scan<BLOCK_SIZE, ELEMS>(g, shm);

      for (int i = 0; i < ELEMS; ++i) {
        int lidx = i * BLOCK_SIZE + lid;
        int gidx = bid * BLOCK_ELEMS + lidx;
        if (gidx < n) {
          d_out[gidx] = shm[lidx];
        }
      }

      if (d_block_sum && g.leader()) {
        d_block_sum[bid] = shm[BLOCK_ELEMS - 1];
      }
    });
  });

  if (!d_block_sum) {
    return e;
  }

  e = recursive_scan_impl<ScanType::Exclusive, BLOCK_SIZE, ELEMS>(
      q, num_groups, d_block_sum, d_block_sum, d_scratch + num_groups, {&e, 1});

  return q.submit([&](sycl::handler &cg) {
    cg.depends_on(e);
    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      auto g = id.get_group();
      int lid = id.get_local_id();
      int bid = g.get_group_id();
      for (int i = 0; i < ELEMS; ++i) {
        int lidx = i * BLOCK_SIZE + lid;
        int gidx = bid * BLOCK_ELEMS + lidx;
        if (gidx < n) {
          d_out[gidx] += d_block_sum[bid];
        }
      }
    });
  });
}

template <ScanType ST>
auto recursive_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
                    std::span<const sycl::event> dependences = {})
    -> sycl::event {
  constexpr int BLOCK_SIZE = 64;
  constexpr int ELEMS = 8;
  constexpr int BLOCK_ELEMS = BLOCK_SIZE * ELEMS;

  if (n == 0) {
    return {};
  }

  size_t scratch_cnt = 0;
  size_t num_groups = n;
  do {
    num_groups = ceil_div(num_groups, BLOCK_ELEMS);
    scratch_cnt += num_groups;
  } while (num_groups > 1);

  int *d_scratch =
      scratch_cnt > 1 ? sycl::malloc_device<int>(scratch_cnt, q) : nullptr;

  sycl::event e = recursive_scan_impl<ST, BLOCK_SIZE, ELEMS>(
      q, n, d_data, d_out, d_scratch, dependences);

  std::thread([q, e, d_scratch]() mutable {
    e.wait();
    sycl::free(d_scratch, q);
  }).detach();

  return e;
}

} // namespace

namespace {

template <ScanType ST>
auto stream_scan(sycl::queue &q, int n, const int *d_data, int *d_out,
                 std::span<const sycl::event> dependences = {}) -> sycl::event {
  constexpr int BLOCK_SIZE = 1024;
  constexpr int ELEMS = 7;
  constexpr int BLOCK_ELEMS = BLOCK_SIZE * ELEMS;
  constexpr int COUNTER_NUM_STARTED = 0;
  constexpr int COUNTER_NUM_FINISHED = 1;

  size_t num_groups = ceil_div(n, BLOCK_ELEMS);

  int *d_atomics = sycl::malloc_device<int>(2, q);
  int *d_per_block_exc_sums = sycl::malloc_device<int>(num_groups + 1, q);

  sycl::event e = q.single_task([=] {
    d_atomics[COUNTER_NUM_STARTED] = 0;
    d_atomics[COUNTER_NUM_FINISHED] = 0;
    d_per_block_exc_sums[0] = 0;
  });

  e = q.submit([&](sycl::handler &cg) {
    constexpr int SHM_ROW_ELEMS = ELEMS + (ELEMS % 2 == 0);

    sycl::local_accessor<int, 2> shm({BLOCK_SIZE, SHM_ROW_ELEMS}, cg);
    sycl::local_accessor<int> scan_shm(BLOCK_SIZE, cg);

    constexpr int SHM_SIZE = BLOCK_SIZE * SHM_ROW_ELEMS + BLOCK_SIZE;

    cg.depends_on(e);
    depends_on(cg, dependences);

    sycl::nd_range<1> range = {num_groups * BLOCK_SIZE, BLOCK_SIZE};
    cg.parallel_for(range, [=](sycl::nd_item<1> id) {
      auto g = id.get_group();
      auto sg = id.get_sub_group();
      int lid = id.get_local_id();
      int sg_lid = sg.get_local_id();

      if (lid == 0) {
        sycl::atomic_ref<int, sycl::memory_order_relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            num_started_ref(d_atomics[COUNTER_NUM_STARTED]);
        int bid = num_started_ref.fetch_add(1);

        scan_shm[BLOCK_SIZE - 1] = bid;
      }
      sycl::group_barrier(g);

      int bid;
      if (sg_lid == 0) {
        bid = scan_shm[BLOCK_SIZE - 1];
      }
      sycl::group_barrier(sg);
      bid = sycl::group_broadcast(sg, bid, 0);

      int r = 0;
      for (int i = 0; i < ELEMS; ++i) {
        int gidx = bid * BLOCK_ELEMS + lid * ELEMS + i;
        int v;
        if constexpr (ST == ScanType::Exclusive) {
          v = gidx > 0 && gidx < n ? d_data[gidx - 1] : 0;
        } else if constexpr (ST == ScanType::Inclusive) {
          v = gidx < n ? d_data[gidx] : 0;
        }
        r += v;
        shm[lid][i] = v;
      }
      scan_shm[lid] = r;
      sycl::group_barrier(g);

      group_inclusive_scan<BLOCK_SIZE, 1>(g, scan_shm);

      if (lid == 0) {
        sycl::atomic_ref<int, sycl::memory_order_relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            num_finished_ref(d_atomics[COUNTER_NUM_FINISHED]);

        while (num_finished_ref.load(sycl::memory_order_acquire) != bid) {
        }
        int per_block_exc_sum = d_per_block_exc_sums[bid];

        int block_sum = scan_shm[BLOCK_SIZE - 1];
        d_per_block_exc_sums[bid + 1] = per_block_exc_sum + block_sum;
        num_finished_ref.store(bid + 1, sycl::memory_order_release);

        scan_shm[BLOCK_SIZE - 1] = per_block_exc_sum;
      }
      sycl::group_barrier(g);

      int per_block_exc_sum;
      if (sg_lid == 0) {
        per_block_exc_sum = scan_shm[BLOCK_SIZE - 1];
      }
      sycl::group_barrier(sg);
      per_block_exc_sum = sycl::group_broadcast(sg, per_block_exc_sum, 0);

      int s = per_block_exc_sum + (lid > 0 ? scan_shm[lid - 1] : 0);
      for (int i = 0; i < ELEMS; ++i) {
        s += shm[lid][i];

        int gidx = bid * BLOCK_ELEMS + lid * ELEMS + i;
        if (gidx < n) {
          d_out[gidx] = s;
        }
      }
    });
  });

  std::thread([q, e, d_atomics, d_per_block_exc_sums]() mutable {
    e.wait();
    sycl::free(d_atomics, q);
    sycl::free(d_per_block_exc_sums, q);
  }).detach();

  return e;
}

} // namespace

auto exclusive_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
                    std::span<const sycl::event> dependences) -> sycl::event {
  return exclusive_stream_scan(q, n, d_data, d_out, dependences);
}

auto inclusive_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
                    std::span<const sycl::event> dependences) -> sycl::event {
  return inclusive_stream_scan(q, n, d_data, d_out, dependences);
}

auto exclusive_recursive_scan(sycl::queue &q, size_t n, const int *d_data,
                              int *d_out,
                              std::span<const sycl::event> dependences)
    -> sycl::event {
  return recursive_scan<ScanType::Exclusive>(q, n, d_data, d_out, dependences);
}

auto inclusive_recursive_scan(sycl::queue &q, size_t n, const int *d_data,
                              int *d_out,
                              std::span<const sycl::event> dependences)
    -> sycl::event {
  return recursive_scan<ScanType::Inclusive>(q, n, d_data, d_out, dependences);
}

auto exclusive_stream_scan(sycl::queue &q, size_t n, const int *d_data,
                           int *d_out, std::span<const sycl::event> dependences)
    -> sycl::event {
  return stream_scan<ScanType::Exclusive>(q, n, d_data, d_out, dependences);
}

auto inclusive_stream_scan(sycl::queue &q, size_t n, const int *d_data,
                           int *d_out, std::span<const sycl::event> dependences)
    -> sycl::event {
  return stream_scan<ScanType::Inclusive>(q, n, d_data, d_out, dependences);
}

} // namespace syclalgo
