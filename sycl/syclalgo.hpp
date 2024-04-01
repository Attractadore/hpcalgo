#pragma once
#include <cstddef>
#include <span>
#include <sycl/sycl.hpp>

namespace syclalgo {

auto saxpy(sycl::queue &q, size_t n, float a, sycl::buffer<float> &x,
           sycl::buffer<float> &y) -> sycl::event;

auto saxpy(sycl::queue &q, size_t n, float a, const float *d_x, float *d_y,
           std::span<const sycl::event> dependences = {}) -> sycl::event;

} // namespace syclalgo
