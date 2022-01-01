cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(testbench LANGUAGES CXX)

add_executable(testbench
    main.cpp
    )

find_package(OpenMP REQUIRED)
target_link_libraries(testbench PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(testbench PUBLIC -ffast-math -march=native)
