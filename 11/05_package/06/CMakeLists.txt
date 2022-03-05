add_executable(main main.cpp)

find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} C:/Qt/Qt5.14.2/msvc2019_64/lib/cmake)

find_package(Qt5 REQUIRED COMPONENTS Widgets Gui REQUIRED)
target_link_libraries(main PUBLIC Qt5::Widgets Qt5::Gui)
