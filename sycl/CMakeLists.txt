cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

if (NOT SYCLALGO_COMPILER)
  find_program(icpx icpx)
  if (icpx)
    message(STATUS "Found icpx: ${icpx}")
    set(SYCLALGO_COMPILER DPCPP)
  endif()
endif()

if (NOT SYCLALGO_COMPILER)
  if (DEFINED ENV{DPCPP_HOME})
    message(STATUS "Found DPCPP_HOME: $ENV{DPCPP_HOME}")
    set(SYCLALGO_COMPILER INTEL_LLVM)
  endif()
endif()

if (NOT SYCLALGO_COMPILER)
  message(STATUS "Using AdaptiveCpp as default SYCL compiler")
  set(SYCLALGO_COMPILER ADAPTIVE_CPP)
endif()

if (SYCLALGO_COMPILER MATCHES ADAPTIVE_CPP)
  message(STATUS "SYCL compiler: AdaptiveCpp")
  enable_language(CXX)
  find_package(AdaptiveCpp REQUIRED CONFIG)
elseif(SYCLALGO_COMPILER MATCHES INTEL_LLVM)
  message(STATUS "SYCL compiler: Intel LLVM")
  find_program(CMAKE_CXX_COMPILER clang++ NO_DEFAULT_PATH PATHS $ENV{DPCPP_HOME}/llvm/build/bin REQUIRED)
  list(APPEND CMAKE_BUILD_RPATH $ENV{DPCPP_HOME}/llvm/build/lib)
elseif(SYCLALGO_COMPILER MATCHES DPCPP)
  message(STATUS "SYCL compiler: Data Parallel C++")
  find_program(CMAKE_CXX_COMPILER icpx REQUIRED)
elseif(NOT SYCLALGO_COMPILER)
  message(FATAL_ERROR "SYCL compiler not found")
else()
  message(FATAL_ERROR "Unexpected SYCL compiler: ${SYCLALGO_COMPILER}")
endif()

project(syclalgo LANGUAGES CXX)

if(SYCLALGO_COMPILER MATCHES INTEL_LLVM OR SYCLALGO_COMPILER MATCHES DPCPP)
  set(dpcpp_flags -fsycl)

  find_package(HIP CONFIG)
  if (HIP_FOUND)
    if (HIP_PLATFORM MATCHES amd)
      list(APPEND dpcpp_flags -fsycl-targets=amdgcn-amd-amdhsa)

      execute_process(COMMAND ${HIP_HIPCONFIG_EXECUTABLE} -R OUTPUT_VARIABLE rocm_path COMMAND_ERROR_IS_FATAL ANY)
      execute_process(COMMAND ${rocm_path}/bin/rocm_agent_enumerator OUTPUT_VARIABLE rocm_agents COMMAND_ERROR_IS_FATAL ANY)
      string(REGEX REPLACE "(\r?\n)+" ";" rocm_agents ${rocm_agents})
      list(REMOVE_ITEM rocm_agents gfx000)
      list(REMOVE_ITEM rocm_agents "")
      foreach (rocm_agent ${rocm_agents})
        list(APPEND dpcpp_flags "SHELL:-Xsycl-target-backend --offload-arch=${rocm_agent}")
      endforeach()
    endif()
  endif()

  function(add_sycl_to_target)
    set(options)
    set(one_value_keywords TARGET)
    cmake_parse_arguments(ADD_SYCL
      "${options}"
      "${one_value_keywords}"
      "${multi_value_keywords}"
      ${ARGN}
    )
    target_compile_options(${ADD_SYCL_TARGET} PRIVATE ${dpcpp_flags})
    target_link_options(${ADD_SYCL_TARGET} PRIVATE ${dpcpp_flags})
    target_compile_definitions(${ADD_SYCL_TARGET} PRIVATE DPCPP)
  endfunction()
endif()

if (SYCLALGO_COMPILER MATCHES DPCPP)
  find_package(MKL CONFIG)
  if (TARGET MKL::MKL_DPCPP)
    target_compile_definitions(MKL::MKL_DPCPP INTERFACE ONEMKL)
  endif()
endif()

find_package(benchmark REQUIRED CONFIG)
find_package(GTest REQUIRED CONFIG)
include(GoogleTest)

add_library(syclalgo syclalgo.cpp)
add_sycl_to_target(TARGET syclalgo)

enable_testing()
add_executable(sycltest sycltest.cpp)
target_link_libraries(sycltest PRIVATE syclalgo GTest::gtest_main)
add_sycl_to_target(TARGET sycltest)
gtest_discover_tests(sycltest)

add_executable(syclbench-saxpy syclbench-saxpy.cpp)
target_link_libraries(syclbench-saxpy PRIVATE syclalgo $<TARGET_NAME_IF_EXISTS:MKL::MKL_DPCPP> benchmark::benchmark_main)
add_sycl_to_target(TARGET syclbench-saxpy)
