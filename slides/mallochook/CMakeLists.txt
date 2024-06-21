cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -rdynamic")
project(mallochook LANGUAGES CXX)

add_compile_options(-Wall -Wextra -Werror=return-type)

add_library(mallochook SHARED hook.cpp plot.cpp)
add_library(leakcheck SHARED check.cpp)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE mallochook)
