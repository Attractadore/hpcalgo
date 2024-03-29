#include "hipalgo.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <numeric>

namespace {

void test_exc_scan(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::exclusive_scan(data.begin(), data.end(), scan.begin(), 0);

  int *d_data;
  HIP_TRY(hipMalloc(&d_data, sizeof(int) * n));
  HIP_TRY(
      hipMemcpy(d_data, data.data(), sizeof(int) * n, hipMemcpyHostToDevice));

  int *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(int) * n));

  hipalgo::exc_scan(n, d_data, d_result);

  std::vector<int> result(n);
  HIP_TRY(hipMemcpy(result.data(), d_result, sizeof(int) * n,
                    hipMemcpyDeviceToHost));

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));

  EXPECT_EQ(scan, result);
}

} // namespace

TEST(Scan, ExcScan) {
  {
    SCOPED_TRACE("exc_scan: single block");
    test_exc_scan(100);
  }
  {
    SCOPED_TRACE("exc_scan: multi block");
    test_exc_scan(1000);
  }
  {
    SCOPED_TRACE("exc_scan: multi level");
    test_exc_scan(100'000);
  }
}

namespace {

void test_inc_scan(size_t n) {
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 1);

  std::vector<int> scan(n);
  std::inclusive_scan(data.begin(), data.end(), scan.begin());

  int *d_data;
  HIP_TRY(hipMalloc(&d_data, sizeof(int) * n));
  HIP_TRY(
      hipMemcpy(d_data, data.data(), sizeof(int) * n, hipMemcpyHostToDevice));

  int *d_result;
  HIP_TRY(hipMalloc(&d_result, sizeof(int) * n));

  hipalgo::inc_scan(n, d_data, d_result);

  std::vector<int> result(n);
  HIP_TRY(hipMemcpy(result.data(), d_result, sizeof(int) * n,
                    hipMemcpyDeviceToHost));

  HIP_TRY(hipFree(d_data));
  HIP_TRY(hipFree(d_result));

  EXPECT_EQ(scan, result);
}

} // namespace

TEST(Scan, IncScan) {
  {
    SCOPED_TRACE("inc_scan: single block");
    test_inc_scan(100);
  }
  {
    SCOPED_TRACE("inc_scan: multi block");
    test_inc_scan(1000);
  }
  {
    SCOPED_TRACE("inc_scan: multi level");
    test_inc_scan(100'000);
  }
}
