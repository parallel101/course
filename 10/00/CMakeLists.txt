cmake_minimum_required(VERSION 3.18)  # Please upgrade CMake, otherwise try remove this line

set(CMAKE_CXX_STANDARD 17)  # Windows please use VS2019, the funny VS2017 doesn't support C++17 at all
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Of course, not using Windows would be the best choice

# Peng-sensei uses: Arch Linux, CMake 3.22, GCC 11.1, CUDA 11.5
# Feel free to contact me in the WeChat group for troubleshooting CMake :)
# To join our WeChat group, add WeChat ID 'janicehasadream' as good-friend
# She will invite you into the WeChat group and sorry for the inconvinence
# The stupid Zhang-xiaolong refuse to give qrcode when member count >=200!

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# When testing performance, please switch to Release build, Windows may use:
# cmake -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WIN32)  # Cihou Wendous
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

project(main LANGUAGES CXX)

add_executable(main 10.cpp)  # Switch to 01.cpp here, to play different examples

find_package(OpenMP)
if (OpenMP_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found")
endif()
