cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(testbench LANGUAGES CXX)

add_executable(testbench
    main.cpp
    aos.cpp
    aos_aligned.cpp
    aos_parallel.cpp
    soa.cpp
    soa_size_t.cpp
    soa_unroll.cpp
    soa_simd.cpp
    soa_parallel.cpp
    aosoa.cpp
    )

find_package(OpenMP REQUIRED)
target_link_libraries(testbench PUBLIC OpenMP::OpenMP_CXX)
if (DEFINED TEST_SIZE)
    target_compile_definitions(testbench PUBLIC N=${TEST_SIZE})
endif()
if (DEFINED TEST_SAMPLES)
    target_compile_definitions(testbench PUBLIC TIMES=${TEST_SAMPLES})
endif()
