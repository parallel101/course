cmake_minimum_required(VERSION 3.12)
project(hellocmake LANGUAGES CXX)

add_subdirectory(fmt)

add_executable(a.out main.cpp)
target_link_libraries(a.out PUBLIC fmt)
