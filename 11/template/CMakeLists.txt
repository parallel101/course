cmake_minimum_required(VERSION 3.15)
project(hellocmake)

if (DEFINED ENV{MYVAR})
    message("MYVAR is: $ENV{MYVAR}")
else()
    message("MYVAR is not defined")
endif()

add_executable(main main.cpp)
