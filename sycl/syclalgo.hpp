#pragma once
#include <cstddef>
#include <span>
#include <sycl/sycl.hpp>

namespace syclalgo {

auto saxpy(sycl::queue &q, std::span<const sycl::event> dependences, size_t n,
           float a, const float *d_x, float *d_y) -> sycl::event;

} // namespace syclalgo
