#pragma once
#include <cstddef>
#include <span>
#include <sycl/sycl.hpp>

namespace syclalgo {

void saxpy(sycl::queue &q, size_t n, float a, sycl::buffer<float> &x,
           sycl::buffer<float> &y);

auto saxpy(sycl::queue &q, size_t n, float a, const float *d_x, float *d_y,
           std::span<const sycl::event> dependences = {}) -> sycl::event;

void exc_scan(sycl::queue &q, size_t n, sycl::buffer<int> &data,
              sycl::buffer<int> &out);

void inc_scan(sycl::queue &q, size_t n, sycl::buffer<int> &data,
              sycl::buffer<int> &out);

auto exc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences = {}) -> sycl::event;

auto inc_scan(sycl::queue &q, size_t n, const int *d_data, int *d_out,
              std::span<const sycl::event> dependences = {}) -> sycl::event;

} // namespace syclalgo
