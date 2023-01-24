cmake_minimum_required(VERSION 3.15)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

project(MapCourse LANGUAGES CXX)

set(DEMO_LIST
    demo0
)

foreach (name IN ITEMS ${DEMO_LIST})
    add_executable(${name} ${name}.cpp)
endforeach()
