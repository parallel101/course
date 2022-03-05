cmake_minimum_required(VERSION 3.15)

set(MYVAR OFF)
add_subdirectory(mylib)
message("MYVAR: ${MYVAR}")

add_executable(main main.cpp)
