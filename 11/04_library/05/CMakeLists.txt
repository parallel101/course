add_library(mylib OBJECT mylib.cpp)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC mylib)
