cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

project(main)

add_compile_options(-Wall -Wextra -Werror=return-type)
if (MSVC)
    add_compile_options(/Zc:preprocessor)
endif()

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
foreach(source IN LISTS sources)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE jsoncpp)
    target_include_directories(${name} PRIVATE /usr/include/python3.12)
    # find_package(pybind11 REQUIRED)
    # target_link_libraries(${name} PRIVATE pybind11)
endforeach()
