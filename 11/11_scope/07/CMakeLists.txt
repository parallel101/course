cmake_minimum_required(VERSION 3.15)
project(hellocmake)

set(MYVAR "")
if (DEFINED MYVAR)
    message("MYVAR is: ${MYVAR}")
else()
    message("MYVAR not defined")
endif()

add_executable(main main.cpp)
