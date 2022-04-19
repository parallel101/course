# 高性能并行编程与优化 - 课件

欢迎光临开源并行课！您将学到现代 C++ 与高性能计算相关知识！
这里用来存放上课用到的 **源代码** 和 **课件** 等。

* 每周六14点开播：https://live.bilibili.com/14248205
* 录播也会上传到：https://space.bilibili.com/263032155

![CC-BY-NC-SA](tools/cc-by-nc-sa.jpg)

# 下载课件

如果你不知道 Git 如何使用，可以点击这里：[一键下载](https://github.com/archibate/course/archive/refs/heads/master.zip)。

# 目录结构

* 01/slides.ppt - 第一课的课件
* 01/01 - 第一课第一小节的代码
* 01/02 - 第一课第二小节的代码
* 02/slides.ppt - 第二课的课件
* 以此类推……

每一小节的代码目录下都有一个 run.sh，里面是编译运行该程序所用的命令。

# 课程大纲

课程分为前半段和后半段，前半段主要介绍现代 C++，后半段主要介绍并行编程与优化。

1. 课程安排与开发环境搭建：cmake与git入门
1. 现代C++入门：常用STL容器，RAII内存管理
1. 现代C++进阶：模板元编程与函数式编程
1. 编译器如何自动优化：从汇编角度看C++
1. C++11起的多线程编程：从mutex到无锁并行
1. 并行编程常用框架：OpenMP与Intel TBB
1. 被忽视的访存优化：内存带宽与cpu缓存机制
1. GPU专题：wrap调度，共享内存，barrier
1. 并行算法实战：reduce，scan，矩阵乘法等
1. 存储大规模三维数据的关键：稀疏数据结构
1. 物理仿真实战：邻居搜索表实现pbf流体求解
1. C++在ZENO中的工程实践：从primitive说起
1. 结业典礼：总结所学知识与优秀作业点评

# 前置条件

硬件要求：
- 64位（32位时代过去了）
- 至少2核4线程（并行课…）
- 英伟达家显卡（GPU 专题）

软件要求：
- Visual Studio 2019（Windows用户）
- GCC 9 及以上（Linux用户）
- CMake 3.12 及以上（跨平台作业）
- Git 2.x（作业上传到 GitHub）
- CUDA Toolkit 10.0 以上（GPU 专题）

# 参考资料

- [C++ 官方文档](https://en.cppreference.com/w/)
- [C++ 核心开发规范](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
- [Effective Mordern C++ 中文版](https://github.com/kelthuzadx/EffectiveModernCppChinese/blob/master/4.SmartPointers/item22.md)
- [热心观众整理的学习资料](https://github.com/jiayaozhang/OpenVDB_and_TBB)
- [LearnCpp 中文版](https://learncpp-cn.github.io/)
- [Performance Analysis and Tuning on Modern CPUs](http://faculty.cs.niu.edu/~winans/notes/patmc.pdf)
- [C++ 并发编程实战](https://www.bookstack.cn/read/Cpp_Concurrency_In_Action/README.md)
- [深入理解计算机原理 (CSAPP)](http://csapp.cs.cmu.edu/)
- [并行体系结构与编程 (CMU 15-418)](https://www.bilibili.com/video/av48153629/)
- [因特尔 TBB 编程指南](https://www.inf.ed.ac.uk/teaching/courses/ppls/TBBtutorial.pdf)
- [CMake “菜谱”](https://www.bookstack.cn/read/CMake-Cookbook/README.md)
- [CMake 官方文档](https://cmake.org/cmake/help/latest/)
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 官方文档](https://docs.github.com/en)
- [助教老师知乎](https://www.zhihu.com/people/AlbertRen/posts)
- [实用网站 CppInsights 解构 C++ 语法糖](https://cppinsights.io)
- [实用网站 GodBolt 查看不同编译器生成的汇编](http://godbolt.org)

