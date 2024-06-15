cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

project(main)

set(SOURCES main.cpp)

add_compile_options(-Wall -Wextra -Werror=return-type)
add_executable(main main.cpp)

find_package(benchmark REQUIRED)
target_link_libraries(main benchmark::benchmark benchmark::benchmark_main)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
