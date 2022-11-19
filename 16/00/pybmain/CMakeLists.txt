file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_executable(pybmain ${srcs})
target_include_directories(pybmain PUBLIC include)

target_link_libraries(pybmain PUBLIC biology)
