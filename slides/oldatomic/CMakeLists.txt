cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 23)

project(main)

set(SOURCES main.cpp)

add_compile_options(-Wall -Wextra -Werror=return-type)
add_executable(main main.cpp)
add_executable(test test.cpp)
add_dependencies(test main)

find_package(TBB REQUIRED)
if (TARGET TBB::tbb)
    target_link_libraries(test PRIVATE TBB::tbb)
    target_compile_definitions(test PRIVATE HAS_TBB)
endif()

add_custom_target(run
    COMMAND $<TARGET_FILE:test>
    DEPENDS test
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
