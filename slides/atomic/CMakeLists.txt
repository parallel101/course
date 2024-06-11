cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 23)

project(main)

set(SOURCES main.cpp)

add_executable(main ${SOURCES})
target_compile_options(main PRIVATE -Wall -Wextra -Werror=return-type)
