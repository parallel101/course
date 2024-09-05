# 现代 C++ 的 CUDA 编程

参考资料：

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://www.cs.sfu.ca/~ashriram/Courses/CS431/assets/lectures/Part8/GPU1.pdf

## 配置 CUDA 开发环境

硬件方面建议使用至少 GTX 1060 以上显卡，但是更老的显卡也可以运行。

软件方面则可以尽可能最新，以获得 CUDA C++20 支持，我安装的版本是 CUDA 12.5。

以下仅演示 Arch Linux 中安装 CUDA 的方法，因为 Arch Linux 官方源中就自带 `nvidia` 驱动和 `cuda` 包，而且开箱即用，其他发行版请自行如法炮制。

Wendous 用户可能在安装完后遇到“找不到 cuxxx.dll”报错，说明你需要拷贝 CUDA 安装目录下的所有 DLL 到 `C:\\Windows\\System32`。

WSL 用户要注意，WSL 环境和真正的 Linux 相差甚远。很多 Linux 下的教程，你会发现在 WSL 里复刻不出来。这是 WSL 的 bug，应该汇报去让微软统一修复，而不是让教程的作者零零散散一个个代它擦屁股。建议直接在 Wendous 本地安装 CUDA 反而比伺候 WSL 随机拉的 bug 省力。

Ubuntu 用户可能考虑卸载 Ubuntu，因为 Ubuntu 源中的版本永不更新。想要安装新出的软件都非常困难，基本只能安装到五六年前的古董软件，要么只能从网上下 deb 包，和 Wendous 一个软耸样。所有官方 apt 源中包的版本从 Ubuntu 发布那一天就定死了，永远不会更新了。这是为了起夜级服务器安全稳定的需要，对于个人电脑而言却只是白白阻碍我们学习，Arch Linux 这样的滚动更新的发行版才更适合个人桌面用户。

### 安装 NVIDIA 驱动

首先确保你安装了 NVIDIA 最新驱动：

```bash
pacman -S nvidia
```

运行以下命令，确认显卡驱动正常工作：

```bash
nvidia-smi
```

应该能得到：

```
Mon Aug 26 14:09:15 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.58.02              Driver Version: 555.58.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 ...    Off |   00000000:01:00.0  On |                  N/A |
|  0%   30C    P8             17W /  285W |     576MiB /  16376MiB |     41%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A       583      G   /usr/lib/Xorg                                 370MiB |
|    0   N/A  N/A       740      G   xfwm4                                           4MiB |
|    0   N/A  N/A       783      G   /usr/lib/firefox/firefox                      133MiB |
|    0   N/A  N/A      4435      G   obs                                            37MiB |
+-----------------------------------------------------------------------------------------+
```

如果不行，那就重启。

### 安装 CUDA

然后安装 CUDA Toolkit（即 nvcc 编译器）：

```bash
pacman -S cuda
```

打开 `.bashrc`（如果你是 zsh 用户就打开 `.zshrc`），在末尾添加两行：

```bash
export PATH="/opt/cuda/bin:$PATH"    # 这是默认的 cuda 安装位置
export NVCC_CCBIN="/usr/bin/g++-13"  # Arch Linux 用户才需要这一行
```

然后重启 `bash`，或者执行以下命令重载环境变量：

```bash
source .bashrc
```

运行以下命令测试 CUDA 编译器是否可用：

```bash
nvcc --version
```

应该能得到：

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
```

### 常见问题解答

CMake 报错找不到 CUDA？添加环境变量：

```bash
export PATH="/opt/cuda/bin:$PATH"    # 这里换成你的 cuda 安装位置
export NVCC_CCBIN="/usr/bin/g++-13"  # 只有 Arch Linux 需要这一行
```

IDE 使用了 Clangd 静态检查插件，报错不认识 `-forward-unknown-to-host-compiler` 选项？

创建文件 `~/.config/clangd/config.yaml`：

```yaml
CompileFlags:
  Add:     # 要额外添加到 Clang 的 NVCC 没有的参数
    - --no-cuda-version-check
  Remove:  # 移除 Clang 不认识的 NVCC 参数
    - -forward-unknown-to-host-compiler
    - --expt-*
    - --generate-code=*
    - -arch=*
    - -rdc=*
```

### 建议开启的 CMake 选项

#### CUDA 编译器路径

如果你无法搞定环境变量，也可以通过 `CMAKE_CUDA_COMPILER` 直接设置 `nvcc` 编译器的路径：

```cmake
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")  # 这里换成你的 cuda 安装位置
```

不建议这样写，因为会让使用你项目的人也被迫把 CUDA 安装到这个路径去。

建议是把你的 `nvcc` 安装好后，通过 `PATH` 环境变量，`cmake` 就能找到了，不需要设置这个变量。

#### CUDA C++ 版本

CUDA 是一种基于 C++ 的领域特定语言，CUDA C++ 的版本和正规 C++ 一一对应。

目前最新的是 CUDA C++20，可以完全使用 C++20 特性的同时书写 CUDA 代码。

- 在 `__host__` 函数（未经特殊修饰的函数默认就是此类，在 CPU 端执行）中，CUDA 和普通 C++ 没有区别，任何普通 C++ 代码，都可以用 CUDA 编译器编译。
- 在 `__device__` 函数（CUDA kernel，在 GPU 端执行）中，能使用的函数和类就有一定限制了：
    - 例如你不能在 `__device__` 函数里使用仅限 `__host__` 用的 `std::cout`（但 `printf` 可以，因为 CUDA 团队为了方便用户调试，为你做了 `printf` 的 `__device__` 版特化）。
    - `__device__` 中不能使用绝大多数非 `constexpr` 的 STL 容器，例如 `std::map` 等，但是在 `__host__` 侧还是可以用的！
    - 所有的 `constexpr` 函数也是可以使用的，例如各种 C++ 风格的数学函数如 `std::max`，`std::sin`，这些函数都是 `constexpr` 的，在 `__host__` 和 `__device__` 都能用。
    - 如果一个容器的成员全是 `constexpr` 的，那么他可以在 `__device__` 函数中使用。例如 `std::tuple`、`std::array` 等等，因为不涉及 I/O 和内存分配，都是可以在 `__device__` 中使用的。
    - 例如 C++20 增加了 constexpr-new 的支持，让 `std::vector` 和 `std::string` 变成了 `constexpr` 的容器，因此可以在 `__device__` 中使用 `std::vector`（会用到 `__device__` 版本的 `malloc` 函数，这是 CUDA 的一大特色：你可以在 kernel 内部用 `malloc` 动态分配设备内存，并且从 CUDA C++20 开始 `new` 也可以了）。
    - `std::variant` 现在也是 `constexpr` 的容器，也可以在 `__device__` 函数中使用了。
    - 异常目前还不是 `constexpr` 的，因此无法在 `__device__` 函数中使用 `try/catch/throw` 系列关键字。
    - 总之，随着，我们可以期待越来越多纯计算的函数和容器能在 CUDA kernel（`__device__` 环境）中使用。

正如 `CMAKE_CXX_STANDARD` 设置了 `.cpp` 文件所用的 C++ 版本，也可以用 `CMAKE_CUDA_STANDARD` 设置 `.cu` 文件所用的 CUDA C++ 版本。

```cmake
set(CMAKE_CXX_STANDARD 20)       # .cpp 文件采用的 C++ 版本是 C++20
set(CMAKE_CUDA_STANDARD 20)      # .cu 文件采用的 CUDA C++ 版本是 C++20
```

### 赋能现代 C++ 语法糖

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
```

* `--expt-relaxed-constexpr`: 让所有 `constexpr` 函数默认自动带有 `__host__ __device__`
* `--expt-extended-lambda`: 允许为 lambda 表达式指定 `__host__` 或 `__device__`

#### 显卡架构版本号

不同的显卡有不同的“架构版本号”，架构版本号必须与你的硬件匹配才能最佳状态运行，可以略低，但将不能发挥完整性能。

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)      # 表示针对 RTX 30xx 系列（Ampere 架构）生成
set(CMAKE_CUDA_ARCHITECTURES native)  # 如果 CMake 版本高于 3.24，该变量可以设为 "native"，让 CMake 自动检测当前显卡的架构版本号
```

架构版本号：例如 75 表示 RTX 20xx 系列（Turing 架构）；86 表示 RTX 30xx 系列（Ampere 架构）；89 表示 RTX 40xx 系列（Ada 架构）等。

完整的架构版本号列表可以在 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) 中找到。

也可以运行如下命令（如果有的话）查询当前显卡的架构版本号：

```bash
__nvcc_device_query
```

#### 设备函数分离定义

默认只有 `__host__` 函数可分离声明和定义。如果你需要分离 `__device__` 函数的声明和定义，就要开启这个选项：

```cmake
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # 可选
```

#### 创建 CUDA 项目

完成以上选项的设定后，使用 `project` 命令正式创建 CUDA C++ 项目。

```cmake
project(这里填你的项目名 LANGUAGES CXX CUDA)
```

> {{ icon.fun }} 我见过有人照抄代码把“这里填你的项目名”抄进去的。

如需在特定条件下才开启 CUDA，可以用 `enable_language()` 命令延迟 CUDA 环境在 CMake 中的初始化：

```cmake
project(这里填你的项目名 LANGUAGES CXX)

...

option(ENABLE_CUDA "Enable CUDA" ON)

if (ENABLE_CUDA)
    enable_language(CUDA)
endif()
```

#### CMake 配置总结

注意！以上这些选项设定都必须在 `project()` 命令之前！否则设定了也无效。

因为实际上是 `project()` 命令会检测这些选项，用这些选项来找到编译器和 CUDA 版本等信息。

总之，我的选项是：

```cmake
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(你的项目名 LANGUAGES CXX CUDA)

file(GLOB sources "*.cpp" "*.cu")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} PRIVATE cusparse cublas)
```

## 开始编写 CUDA

CUDA 有两套 API：

- [CUDA runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)：更加简单，兼顾性能，无需手动编译 kernel，都替你包办好了，但不够灵活。
- [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)：更加灵活多变，但操作繁琐，需要手动编译 kernel，适合有特殊需求的用户。

他们都提供了大量用于管理 CUDA 资源和内存的函数。

我们要学习的是比较易懂、用的也最多的 CUDA runtime API。

使用 `<cuda_runtime.h>` 头文件即可导入所有 CUDA runtime API 的函数和类型：

```cuda
#include <cuda_runtime.h>
```

虽然 CUDA 基于 C++（而不是 C 语言），支持所有 C++ 语言特性。但其 CUDA runtime API 依然是仿 C 风格的接口，可能是照顾了部分从 C 语言转过来的土木老哥，也可能是为了方便被第三方二次封装。

我们的课程主题是：用现代 C++ 赋能更好的 CUDA 开发，所以会对 CUDA 原生的 C 风格 API 做一些 C++ 封装，使其呈更直观易用的接口，帮助你避免出错（例如内存泄漏）。

### 认识 CUDA 语言

一份 CUDA 源码和 C/C++ 一样，是由大量的函数组成。

由于 GPU 编程的特殊性，GPU 代码和 CPU 代码是需要分离的，他们的指令集完全不同。

CPU 的责任是决定什么时候要“启动(launch)” GPU 代码，而 GPU 只专注于计算。

所以 CPU 的指令集中含有大量的条件和判断，而 GPU 则以计算指令为主。

而 C++ 中，函数就是代码，根据代码执行的位置不同，函数可以分为：

- GPU 端执行。
- CPU 端执行。

为此，就需要对进行标识：

CUDA 中的函数分为三大类：

- `__host__` 函数：在 CPU 端执行，只能被 CPU 端函数调用。编译器编译时，会将其编译为 CPU 的汇编（x86 汇编），可以使用所有 C++ 标准库功能（如 `std::cout`）。
- `__device__` 函数：在 GPU 端执行，只能被 GPU 端函数调用。编译器编译时，会将其编译为 GPU 的汇编（PTX 汇编），只能使用 C++ 标准库中纯计算的部分功能（如 `std::sin`）。
- `__global__` 函数：也是在 GPU 端执行，类似于 `__device__`。区别在于：
    - `__global__` 函数可以被 CPU 端的函数（`__host__`）调用。
    - `__device__` 只能被 GPU 端的函数（`__device__` 或 `__global__`）调用。

因而 `__global__` 就像一座桥梁一样，是从 CPU 走向 GPU 的入口点。

> {{ icon.tip }} 但反过来，GPU 函数不能再走回 CPU。所以 `__global__` 是单向的一次性桥梁，一旦进入 GPU，就只能等整个 `__global__` 函数退出了。

`__global__` 就像 GPU 版的 `main` 函数一样，是所有 GPU 代码的入口点。

只不过这个入口点可以有很多个，CPU 可以多次提交不同的 `__global__` 函数，就像是创建了许多个不同的“进程”，分别有各自的 `main` 入口点一样。

这样的一次 `__global__` 调用所产生的一个“GPU 版进程”，被称作一个“网格(grid)”。

正如 CPU 上的单个进程由很多子线程组成一样；GPU 上的每个“网格”，由许多“块(block)”组成，“块”又进一步由许多“线程(thread)”组成。

> {{ icon.detail }} 实际上，GPU 的“块”才最接近 CPU 上线程概念，而 GPU 所谓的“线程”实际上对应于 CPU 的 SIMD 矢量，稍后会详细介绍。

因为很多时候，人们喜欢直接在 `__global__` 函数里写上所有的计算代码，而不会再定义一个 `__device__` 函数了，所有的计算核心功能都在 `__global__` 函数中。

“网格(grid)”是“内核(kernel)”的实例，正如 CPU 上进程是可执行文件(exe)的实例一样：同一个可执行文件可以被 Shell 多次调用产生多个进程，同一个“内核”也可以被 host 函数多次调用产生多个“网格”。

但是，CPU 上的每个进程只会调用一次 `main` 入口点，而 GPU 上的 `__global__` 会被调用 n 次（n 的大小在 host 函数中指定），所有启动的 n 个 `__global__` 函数互相之间是并行执行的，每个线程的入口点都是 `__global__`，因此一个“网格”含有多个“线程”。

```cuda
#include <cuda_runtime.h>

/*__host__*/ void host_func() {
}

__device__ void device_func() {
}

__host__ __device__ void host_device_func() {
}

constexpr void constexpr_func() {
}

__global__ void kernel() {
    device_func();
    host_device_func();
    constexpr_func(); // 需开启 --expt-relaxed-constexpr
    auto device_lambda = [] __device__ (int i) { // 需开启 --expt-extended-lambda
        return i * 2;
    };
    device_lambda(1);
}

int main() {
    host_func();
    host_device_func();
    constexpr_func();
    auto host_lambda = [] (int i) {
        return i * 2;
    };
    host_lambda(1);
}
```
