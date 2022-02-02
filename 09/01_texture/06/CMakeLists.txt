cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(hellocuda LANGUAGES CXX CUDA)

add_executable(main main.cu)
target_include_directories(main PUBLIC . ../../include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(OpenVDB REQUIRED)
message(STATUS "Found OpenVDB ${OpenVDB_VERSION} at ${OpenVDB_LIBRARIES}")
target_link_libraries(main PUBLIC OpenVDB::openvdb)
target_sources(main PUBLIC writevdb.cpp)
