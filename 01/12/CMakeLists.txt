cmake_minimum_required(VERSION 3.12)
project(hellocmake LANGUAGES CXX)

add_executable(a.out main.cpp)
target_include_directories(a.out PUBLIC glm/include)
