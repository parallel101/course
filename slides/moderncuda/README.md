# 现代 C++ 的 CUDA 编程

## 安装 NVIDIA 驱动

首先确保你安装了 NVIDIA 最新驱动：

```bash
pacman -S nvidia
```

运行以下命令，确认显卡驱动正常工作：

```bash
nvidia-smi
```

如果不行，那就重启。

## 安装 CUDA

然后安装 CUDA Toolkit（即 nvcc 编译器）：

```bash
pacman -S cuda
打开 `.bashrc`（如果你是 zsh 用户就打开 `.zshrc`），在末尾添加两行：

```bash
export PATH="/opt/cuda/bin:$PATH"    # 这是默认的 cuda 安装位置
export NVCC_CCBIN="/usr/bin/g++-13"  # Arch Linux 用户才需要这一行
```
## 常见问题

CMake 报错找不到 CUDA？添加环境变量：

```bash
export PATH="/opt/cuda/bin:$PATH"    # 这里换成你的 cuda 安装位置
export NVCC_CCBIN="/usr/bin/g++-13"  # 只有 Arch Linux 需要这一行
```

Clangd 报错不认识 `-forward-unknown-to-host-compiler` 选项？

创建文件 `~/.config/clangd/config.yaml`：

```yaml
CompileFlags:
  Add:
    - --cuda-gpu-arch=sm_86
  Remove:
    - -forward-unknown-to-host-compiler
    - --expt-relaxed-constexpr
    - --expt-extended-lambda
    - --generate-code=*
    - -arch=*
```

### 建议开启的选项

```bash
nvcc --expt-relaxed-constexpr --expt-extended-lambda
```

* --expt-relaxed-constexpr: 让所有 constexpr 函数都自动带有 __host__ __device__
* --expt-extended-lambda: 允许 lambda 表达式具有 __host__ 和/或 __device__ 修饰

### 建议开启的 CMake 选项

```cmake
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 86)      # 不同的显卡有不同的“架构版本号”，架构版本号必须与你的硬件匹配才能最佳状态运行，可以略低，但将不能发挥完整性能。
set(CMAKE_CUDA_ARCHITECTURES native)  # 如果你有 CMake 3.24 以上，可以设定本参数，让 CMake 自动检测当前显卡，并选择准确的架构版本号。
```

架构版本号：例如 75 表示 RTX 20xx 系列（Turing 架构）；86 表示 RTX 30xx 系列（Ampere 架构）；89 表示 RTX 40xx 系列（Ada 架构）等。

完整的架构版本号列表可以在 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) 中找到。

我的选项是：

```cmake
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
```
