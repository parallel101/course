cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(main)

set(SOURCES main.cpp)

add_compile_options(-Wall -Wextra -Werror=return-type)
add_compile_options(-march=native)

add_executable(main main.cpp test.cpp)

find_package(benchmark REQUIRED)
target_link_libraries(main benchmark::benchmark)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
