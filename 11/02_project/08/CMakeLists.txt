cmake_minimum_required(VERSION 3.15)
project(hellocmake
    DESCRIPTION "A free, open-source, online modern C++ course"
    HOMEPAGE_URL https://github.com/parallel101/course
    )

message("PROJECT_NAME: ${PROJECT_NAME}")
message("PROJECT_DESCRIPTION: ${PROJECT_DESCRIPTION}")
message("PROJECT_HOMEPAGE_URL: ${PROJECT_HOMEPAGE_URL}")
add_executable(main main.cpp)
