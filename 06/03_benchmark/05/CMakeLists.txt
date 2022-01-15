cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(main LANGUAGES CXX)

add_executable(main main.cpp)

#find_package(OpenMP REQUIRED)
#target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test!")
add_subdirectory(benchmark)
target_link_libraries(main PUBLIC benchmark)
