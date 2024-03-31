cmake_minimum_required(VERSION 3.21)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

if (NOT SYCLALGO_COMPILER)
  set(ACPP_TARGETS "" CACHE STRING "" FORCE)
  find_package(AdaptiveCpp CONFIG)
  if (AdaptiveCpp_FOUND)
    set(SYCLALGO_COMPILER ADAPTIVE_CPP)
  endif()

  if (NOT SYCLALGO_COMPILER)
    if (DEFINED ENV{DPCPP_HOME})
      message(STATUS "DPCPP_HOME: $ENV{DPCPP_HOME}")
      set(SYCLALGO_COMPILER INTEL_LLVM)
    endif()
  endif()
endif()

if (SYCLALGO_COMPILER MATCHES ADAPTIVE_CPP)
  find_package(AdaptiveCpp REQUIRED CONFIG)
  message(STATUS "SYCL compiler: AdaptiveCpp")
elseif(SYCLALGO_COMPILER MATCHES INTEL_LLVM)
  find_program(CMAKE_CXX_COMPILER clang++ NO_DEFAULT_PATH PATHS $ENV{DPCPP_HOME}/llvm/build/bin REQUIRED)
  message(STATUS "SYCL compiler: Intel LLVM")
  list(APPEND CMAKE_BUILD_RPATH $ENV{DPCPP_HOME}/llvm/build/lib)
else()
  message(FATAL_ERROR "Unexpected SYCL compiler: ${SYCLALGO_COMPILER}")
endif()

project(syclalgo LANGUAGES CXX)

if(SYCLALGO_COMPILER MATCHES INTEL_LLVM)
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
target_link_libraries(syclbench-saxpy PRIVATE syclalgo benchmark::benchmark_main)
add_sycl_to_target(TARGET syclbench-saxpy)
