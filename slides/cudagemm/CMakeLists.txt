cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
    CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(cloth LANGUAGES CXX CUDA)

add_executable(cloth cloth.cu)
target_link_libraries(cloth PRIVATE cusparse cublas)
