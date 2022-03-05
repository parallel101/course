add_executable(main main.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)        # 正确
target_compile_options(main PUBLIC "-std=c++17")          # 错误
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75)  # 正确
target_compile_options(main PUBLIC "-arch=sm_75")         # 错误
