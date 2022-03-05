cmake_minimum_required(VERSION 3.99)
project(hellocmake LANGUAGES CXX)

message("CMAKE_VERSION: ${CMAKE_VERSION}")
add_executable(main main.cpp)
