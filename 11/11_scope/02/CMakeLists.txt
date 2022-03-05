cmake_minimum_required(VERSION 3.15)

add_subdirectory(mylib)
message("MYVAR: ${MYVAR}")

add_executable(main main.cpp)
