cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)

project(cpptest LANGUAGES CXX)

add_executable(cpptest main.cpp)

find_package(Threads REQUIRED)
target_link_libraries(cpptest PUBLIC Threads::Threads)
