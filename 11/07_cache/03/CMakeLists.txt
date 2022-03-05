cmake_minimum_required(VERSION 3.15)
project(hellocmake LANGUAGES CXX)

add_executable(main main.cpp)

set(WITH_TBB ON CACHE BOOL "set to ON to enable TBB, OFF to disable TBB.")
if (WITH_TBB)
    target_compile_definitions(main PUBLIC WITH_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(main PUBLIC TBB::tbb)
endif()
