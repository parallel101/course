cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 14)

project(main)

if (MSVC)
    add_compile_options(/Zc:preprocessor /utf-8 /DNOMINMAX)
else()
    add_compile_options(-Wall -Wextra -Werror=return-type)
endif()

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
foreach(source IN LISTS sources)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
endforeach()
