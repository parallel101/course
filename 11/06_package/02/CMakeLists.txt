add_executable(main main.cpp)

find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)
