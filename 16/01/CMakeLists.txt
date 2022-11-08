cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

project(main LANGUAGES CXX)

add_executable(main main.cpp)

find_package(OpenCV REQUIRED COMPONENTS core videoio)
target_link_libraries(main PUBLIC OpenCV::core)
target_link_libraries(main PUBLIC OpenCV::videoio)
