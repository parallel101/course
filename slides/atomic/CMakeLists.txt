cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 23)

project(main)

set(SOURCES main.cpp)

add_compile_options(-Wall -Wextra -Werror=return-type)
add_executable(main main.cpp)

add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
