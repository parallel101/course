cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(hellocmake LANGUAGES C CXX)

set(Hello world)
set(MYVAR Hello)
if ("${MYVAR}" MATCHES "Hello")
    message("MYVAR is Hello")
else()
    message("MYVAR is not Hello")
endif()

add_executable(main main.cpp)
