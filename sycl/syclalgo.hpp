#pragma once
#include <cstddef>
#include <span>
#include <sycl/sycl.hpp>

namespace syclalgo {

auto saxpy(sycl::queue &q, size_t n, float a, const float *d_x, float *d_y,
           std::span<const sycl::event> dependences = {}) -> sycl::event;

auto exc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences = {}) -> sycl::event;

auto inc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences = {}) -> sycl::event;

} // namespace syclalgo
