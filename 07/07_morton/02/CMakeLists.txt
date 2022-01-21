cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(main LANGUAGES CXX)

add_executable(main main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

#find_package(TBB)
#if (NOT TARGET TBB::tbb)
    #message(WARNING "TBB not found")
#else()
    #target_link_libraries(main PUBLIC TBB::tbb)
    #target_compile_definitions(main PUBLIC -DWITH_TBB)
#endif()

find_package(benchmark REQUIRED)
target_link_libraries(main PUBLIC benchmark::benchmark)

if (MSVC)
    target_compile_options(main PUBLIC /fp:fast /arch:AVX)
else()
    target_compile_options(main PUBLIC -ffast-math -march=native -Wno-narrowing)
endif()
