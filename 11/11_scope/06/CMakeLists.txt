cmake_minimum_required(VERSION 3.15)
project(hellocmake)

message("CMAKE_BUILD_TYPE is: ${CMAKE_BUILD_TYPE}")

add_executable(main main.cpp)
