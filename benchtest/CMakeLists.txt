cmake_minimum_required(VERSION 3.15)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

project(MyBench LANGUAGES CXX)

add_subdirectory(benchmark)

add_executable(demo1 demo1.cpp)
target_link_libraries(demo1 PRIVATE benchmark)
