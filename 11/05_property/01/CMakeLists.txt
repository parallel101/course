add_executable(main main.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)           # 采用 C++17 标准进行编译（默认 11）
set_property(TARGET main PROPERTY CXX_STANDARD_REQUIRED ON)  # 如果编译器不支持 C++17，则直接报错（默认 OFF）
set_property(TARGET main PROPERTY WIN32_EXECUTABLE ON)       # 在 Windows 系统中，运行时不启动控制台窗口，只有 GUI 界面（默认 OFF）
set_property(TARGET main PROPERTY LINK_WHAT_YOU_USE ON)      # 告诉编译器不要自动剔除没有引用符号的链接库（默认 OFF）
set_property(TARGET main PROPERTY LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)   # 设置动态链接库的输出路径（默认 ${CMAKE_BINARY_DIR}）
set_property(TARGET main PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)   # 设置静态链接库的输出路径（默认 ${CMAKE_BINARY_DIR}）
set_property(TARGET main PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)   # 设置可执行文件的输出路径（默认 ${CMAKE_BINARY_DIR}）
