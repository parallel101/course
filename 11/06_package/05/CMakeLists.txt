cmake_minimum_required(VERSION 2.10)
project(hellocmake CXX)

add_executable(main main.cpp)

find_package(TBB COMPONENTS tbb tbbmalloc tbbmalloc_proxy REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
