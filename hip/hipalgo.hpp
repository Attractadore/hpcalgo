#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hipalgo {

#define HIP_TRY(...)                                                           \
  do {                                                                         \
    hipError_t res = __VA_ARGS__;                                              \
    if (res != HIP_SUCCESS) {                                                  \
      std::fprintf(stderr, "%s:%d, %s failed: %d (%s)\n", __FILE__, __LINE__,  \
                   #__VA_ARGS__, res, hipGetErrorName(res));                   \
      std::exit(-1);                                                           \
    }                                                                          \
  } while (0)

void saxpy(size_t n, float a, const float *d_x, float *d_y);

void exc_scan(size_t n, const int *d_data, int *d_out);

void inc_scan(size_t n, const int *d_data, int *d_out);

} // namespace hipalgo
