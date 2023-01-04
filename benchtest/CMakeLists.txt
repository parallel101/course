cmake_minimum_required(VERSION 3.15)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

project(MyBench LANGUAGES CXX)

add_executable(MyBench test.cpp)  # your code

add_subdirectory(benchmark)       # google benchmark
target_link_libraries(MyBench PRIVATE benchmark)
