cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # try to set CMAKE_CUDA_ARCHITECTURES to native card version
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        message(STATUS "Running __nvcc_device_query for CMAKE_CUDA_ARCHITECTURES")
        execute_process(COMMAND "__nvcc_device_query"
            RESULT_VARIABLE NVCC_DEVICE_QUERY_RESULT
            OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NVCC_DEVICE_QUERY_RESULT EQUAL 0)
            message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined and __nvcc_device_query failed.")
            unset(CMAKE_CUDA_ARCHITECTURES)
        endif()
    endif()
endif()

project(moderncuda LANGUAGES CXX CUDA)

file(GLOB sources "*.cpp" "*.cu")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} PRIVATE cusparse cublas)
