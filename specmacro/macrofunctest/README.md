# C++ 初学者常见问题解答

以下为 小彭老师.exe 的源码逆向工程拆包：

我是一个高性能 C++ 教师，当同学提出以下模板问题，我会按模板回答：

## 段错误（崩溃）

Debug 没问题，Release 却崩溃了或结果不对（通常是由于你触发了未定义行为，试试看保守优化模式 -Og 还会不会出错，这种优化后才发作的未定义行为包括：数组访问越界，使用带边界检查的 at() 代替 []；滥用 reinterpret_cast，违背 strict-alising 原则，试试看 -fno-strict-alising 选项还出错不；有符号整数加法/减法/乘法溢出，试试溢出没有 UB 的无符号整数；使用未初始化变量，建议用上节课介绍的 auto-idiom；有返回类型的函数缺少返回语句，试试 -Werror=return-type 看看有没有遗漏）BV17c411G7Y2

有概率段错误，有时又能正常运行，不是 100% 可复现（可能由于多线程在 vector，map，queue 等容器上发生数据竞争引起，请给多个线程同时读写访问的数据结构和类加 mutex 保护）BV1Ya411q7y4

使用二分 print 法定位不到奔溃点（Surprise: 未定义行为可以穿越时间，建议用二分 return 法或二分 throw 法试试；如果能用调试器那最好，例如控制台运行 gdb ./myapp，输入 r 开始运行，然后等到出 SIGSEGV 他会自动暂停，此时输入 bt 即可查看栈回溯，从上往下看）BV1kP4y1K7Eo

段错误，并且你不提供任何有用信息（滥用引用和原始指针，尝试原始指针全部改用智能指针 shared_ptr，new 改用 make_shared，不用 delete）BV1LY411H7Gg

使用了个 lambda，lambda 是延迟执行的，在 lambda 附近发生了段错误（应该是由于你捕获了栈局部变量，而 lambda 的执行在函数退出后，就通过过期的引用访问了已经销毁的栈上变量，导致段错误或得到未定义值，解决方法：把 [&] 改成 [=] 试试，需要浅拷贝的捕获用 shared_ptr 保管后用 [=] 捕获即可保证不产生空悬引用）BV1ui4y1R78s

滥用 string_view 产生空悬引用导致段错误（你踏马会不会生命周期分析钠？不会就给我乖乖用 string，安全总比性能重要，先保证程序能运行了，再去用“chrono测时间”，“intel-vtune profiler”等工具确定瓶颈就在这里了，再去优化，而不是盲目优化浪费精力）BV1ja411M7Di

以上错误排除后，依然存在段错误（你是不是有个函数没写 return？C++ 标准是允许返回类型不为 void 的函数没有 return 语句的，最坑的是他编译期不会报错，二十会造成未定义行为，如果返回类型是一个类，往往会造成段错误，而你还没注意到漏写了个 return，因此我始终开启 -Werror=return-type 选项，MSVC 应该也有类似的开关，这样一旦你漏了 return，他就会立即编译期报错，而不是不痛不痒的警告）

玩指针玩越界玩出 BUG 来了（请看我的 C 语言指针专题视频）BV1US4y1U7Mh

## 静默错误

map 或 unordered_map 访问成员用方括号 m["key"]，没有出错，读到了 0 或空字符串（用 [] 读取元素时，找不到键不会出错而是默默返回 0，还会自动帮你创建那个键，非常危险！建议用找不到就立即报错的 at()，响亮的报错好过静默的出错，只有写入新元素时用 []）

我 cudaDeviceSynchronize 了，没有错误，没有产生正确的结果，printf 也没效果，就好像整个 kernel 没有执行一样，完全静默（你需要用 checkCudaErrors(cudaDeviceSynchronize())，发生这种情况一般是你的 kernel 里发生段错误了，但由于 CUDA 的尿性，段错误只会体现在 cudaDeviceSynchronize 的返回值里，而不会有任何报错信息给你打印在终端上，不仅 cudaDeviceSynchronize 如此，所有 cudaMemcpy，cudaMalloc 函数都是如此，你必须给每个 cudaXXX 函数调用加上 checkCudaErrors 宏，检查一下返回值是否非 0 了，你不加他出错了也不提醒你一下，恶心吧？这就是 CUDA，听黄仁勋说。checkCudaErrors 在 CUDA 官方附赠的 samples 里的 helper_cuda.h 里有，详情请看我的 CUDA 课）BV16b4y1E74f

我调用 glXXXXX 函数没有出错，也没有产生正确的结果（OpenGL 和 CUDA 一个尿性，你的每个 glXXXXX(x, y, z) 函数都必须改成 CHECK_GL(glXXXXX(x, y, z))，是的，每一个都要加，不然踏马出错了也不会提醒你一声，这就是 GL，听 KHR 说）https://github.com/zenustech/zeno/blob/120d92b1f979e809abd24300d78a164607c2f1b0/zenovis/include/zenovis/opengl/common.h#L51

vector<Base> 没有调用派生虚函数（这是由于对象发生了切片 (object-slicing)，push_back(derived) 会退化成 Base 类对象，使用 vector<Base *> 或 vector<shared_ptr<Base>> 代替）

算法正常执行，但得到了错误的结果，或有概率得到错误结果（可能是由于未初始化变量，也可能是多线程数据竞争导致，前者使用上节课讲的 auto-idiom，后者使用 atomic 或 mutex 保护，不要相信脑内内存序）BV1xh4y1d7uh

Derived 是 Base 的子类，Derived 的一些成员变量发生了内存泄漏（Base 的析构函数需要声明为虚函数，virtual ~Base() = default，后面所有继承他的子类都会自动重写这个析构函数）

## 模板报错

使用了 decltype(a)，各种莫名其妙报错，is_same_v 判断无效（因为 decltype(a) 会返回 A const & 而不是 A，用 std::decay_t<decltype(a)> 即可）

T::value_type 无法编译通过（由于缺乏 typename 前缀，用 typename T::value_type 即可）

t.func<U>() 无法编译通过（由于缺乏 template 修饰，用 t.template func<U>() 即可）

出现了一段超长的报错，我给小彭老师截图了，截的是最后几行（我知道你很急，你先别急，你截的图是最后几行，然而这种报错一般都是有一个源头，重点在于那个源头，而由于终端喜欢往下滚动的尿性，你几乎总是会看到最后几行莫名其妙的报错，而你真正要看的是“第一个报错”，只有第一个报错是有价值的，往上滚动，看下第一个报错是什么吧，如果有恼人的警告，那就 -w 参数关闭警告，方便你看到真正有价值的报错）


## 编译失败

GCC/Clang 链接报错：undefined reference to std::__cxx11::basic_string（你正在使用 gcc/clang，请改用 g++/clang++）

MyClass::func 未定义引用（说明这个函数你只声明了却没有实现，试试将分号替换为 {}）

MSVC 链接报错：巴拉巴拉 MT_DynamicRelease 巴拉巴拉（msvc 把 Debug 和 Release 视为是不同的 config，建议统一用 Release 模式：cmake -B build -DCMAKE_BUILD_TYPE=Release; cmake --build build --config Release，此外还必须始终使用静态库，Wendous 的动态库需要一系列 __declspec 伺候，你不会想用的）

Linux 链接静态库到动态库出错（编译那个静态库时需要开启 -fPIC 选项，如果是 CMake 则在最前面添加 set(CMAKE_POSITION_INDEPENDENT_CODE ON) 即可，Linux 建议全部使用动态库，Wendous 建议全部用静态库）

## 性能问题

我的算法太慢了，我是 Python 转过来的（通常是由于 O(N^2) 复杂度，例如在循环中使用 list.index 或字符串拷贝,建议用 map、unordered_map 等高效的关联数据结构，用 std::move(str) 避免大字符串的拷贝）

我的算法太慢了，我喜欢 Java 面向对象（通常是由于滥用 OOP，例如在瓶颈处使用 CString 或虚函数，建议用 data-oriented 编程范式，改用 SOA 或 AOSOA 等对 SIMD 友好的数据布局，以便编译器自动优化）BV12S4y1K721

xxx 和 yyy，哪个更快？（这么急？你实际用 chrono 测量了时间吗？没有测量就没有基准，也不强求你用 Google benchmark 等专业基准测试框架了，连最简单的时间都没测过就开始心理作用优化？）

为什么我用了 OpenMP 并没有变快多少，甚至反而变慢了？（先看看你的数据量是不是足够大，大于 256 KB，总执行时间是否足够长，大于 0.1 ms，否则就是高射炮打蚊子，并行没有意义，如果是处理网络实时收发包的数据，建议先把数据堆到一定量以后再批量并行处理：先等 256 个蚊子全粘在苍蝇板上，再一次性用高射炮处决，这样才高效，是吧？而不是发现一个就立马打掉）

为什么我用了 OpenMP 并没有变快多少，甚至反而变慢了？我的数据量足够大，远大于 256 KB（我们的个人电脑大多是 SMP 架构，并行的“拷贝”并不会比串行的更快，并行加速的是计算，而不能加速访存，对于接近于 memcpy 的 memory-bound（内存瓶颈型）任务，你应该先考虑的是优化访存，而不是急着并行）BV1gu41117bW

OpenMP 如何并行化多层嵌套 for？（如果是 2 层循环就用 #pragma omp parallel for collapse(2)，夹个私货，我个人推荐功能更全的 TBB 而不是恼人的 OpenMP）BV1gu411m7kN

## 灵活性问题

map<K, V> 或 vector<V> 的值类型只能是固定的 V，如何支持任意类型？（使用 std::any 代替固定的 V）

如何实现 variant 内部值的自动获取？auto val = std::get<v.index()>(v) 这样？（这会编译出错，一个函数的返回值，即便用了 auto，也无法同时是多个类型，但是 lambda 函数的 auto 参数可以！std::visit([&] (auto val) { /* 后续代码 */ }, v); 即可解决，prefer callbacks than return values）BV1pa4y1g7v6

如何根据名字，一个动态字符串，访问一个 C++ 类的成员或成员函数？（必应搜索“C++动态反射库”或“C++静态反射库”）

## CMake 问题

我设置了 set(CMAKE_CXX_FLAGS "-std=c++17") 但是没有效果（谁让你这样设置版本的？我课上说了多少遍要用 set(CMAKE_CXX_STANDARD 17) 知道吗？你那个东西有概率会被覆盖而且不跨平台！CMAKE_CXX_STANDARD 才是标准做法）BV1fa411r7zp

我设置了 set(CMAKE_CXX_FLAGS "-O3") 但是没有效果（谁让你这样设置开关优化的？我课上说了多少遍要用 set(CMAKE_BUILD_TYPE Release) 知道吗？你那个东西有概率会被覆盖根本不跨平台！CMAKE_BUILD_TYPE 才是标准做法，Release 开优化，Debug 关优化，RelWithDebInfo 会开优化但保留调试信息，如果要让 Release 从默认的 -O3 变成 -O2 应该用 set(CMAKE_CXX_FLAGS_RELEASE "-O2") 同理还有 CMAKE_CXX_FLAGS_DEBUG 可以设）

我想要链接 pthread，有没有现代 CMake 跨平台的写法？（用 find_package(Threads REQUIRED) 和 target_link_libraries(你的目标程序名 Threads::Threads) 即可，Linux 上等同于 pthread，在 Wendous 上会变成 Wendous 的线程库）

我想要链接某某库，能不能把他做成 subdir 方便我链接？（只有官方支持 CMake subdir 邪教的那些库，如 fmt, spdlog 等可以，否则必须先 make install 那个库到系统中去，然后 find_package 他，subdir 是邪教，不是官方推荐的方法，小彭老师能在 Zeno 里大量用是因为他是 CMake 专家，他有那个本事，如果希望实现完美的“自包含”也可以尝试一下 CMake 自带的 FetchContent 功能，指定一个 URL，他会自动帮你从网上下载依赖项的源码并构建，并使你的主项目能够找到他）BV16P4y1g7MH
