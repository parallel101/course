if (NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

add_library(mylib mylib.cpp)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC mylib)
