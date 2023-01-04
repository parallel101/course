cmake_minimum_required(VERSION 3.15)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

project(BenchCourse LANGUAGES CXX)
add_subdirectory(benchmark)

set(DEMO_LIST
    demo0
)

foreach (name IN ITEMS ${DEMO_LIST})
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE benchmark)
endforeach()
