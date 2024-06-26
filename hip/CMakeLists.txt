cmake_minimum_required(VERSION 3.21)
project(hipalgo LANGUAGES CXX HIP)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(hip REQUIRED CONFIG)
find_package(hipblas REQUIRED CONFIG)
find_package(hipcub REQUIRED CONFIG)
find_package(GTest REQUIRED CONFIG)
find_package(benchmark REQUIRED CONFIG)
include(GoogleTest)

add_library(hipalgo hipalgo.cpp)
set_source_files_properties(hipalgo.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipalgo PUBLIC hip::host)

enable_testing()
add_executable(hiptest hiptest.cpp)
target_link_libraries(hiptest PRIVATE hipalgo GTest::gtest_main)
gtest_discover_tests(hiptest)

add_executable(hipbench-scan hipbench-scan.cpp)
set_source_files_properties(hipbench-scan.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipbench-scan PRIVATE hipalgo hip::hipcub benchmark::benchmark_main)

add_executable(hipbench-saxpy hipbench-saxpy.cpp)
target_link_libraries(hipbench-saxpy PRIVATE hipalgo roc::hipblas benchmark::benchmark_main)
