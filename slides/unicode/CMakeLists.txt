cmake_minimum_required(VERSION 3.10)

project(main LANGUAGES CXX)

file(GLOB sources CONFIGURE_DEPENDS "*.cpp")
foreach(source IN LISTS sources)
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
endforeach()
