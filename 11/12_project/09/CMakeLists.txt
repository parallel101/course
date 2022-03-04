cmake_minimum_required(VERSION 3.15...3.20)
project(hellocmake LANGUAGES CXX)

message("CMAKE_VERSION: ${CMAKE_VERSION}")
message("CMAKE_MINIMUM_REQUIRED_VERSION: ${CMAKE_MINIMUM_REQUIRED_VERSION}")
add_executable(main main.cpp)
