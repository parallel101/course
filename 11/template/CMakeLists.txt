cmake_minimum_required(VERSION 3.15)
project(hellocmake)

message("PATH is: $ENV{PATH}")

add_executable(main main.cpp)
