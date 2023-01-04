cmake_minimum_required(VERSION 3.5)

project(JOMT LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_options(/W3)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG /DQT_NO_DEBUG_OUTPUT")
else()
  add_compile_options(-Wall -Wextra -pedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG -DQT_NO_DEBUG_OUTPUT -march=native -mtune=native")
endif()

# Build the target
add_subdirectory(src)
