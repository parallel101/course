cmake_minimum_required(VERSION 3.15)
project(hellocmake)

set(MYVAR "")
if (MYVAR)
    message("MYVAR is: ${MYVAR}")
else()
    message("MYVAR is empty or not defined")
endif()

add_executable(main main.cpp)
