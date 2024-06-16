#include <benchmark/benchmark.h>

// 问题：内存对齐提高效率？
// 省流：内存对齐并不影响效率，跨缓存行访问才是问题所在

// jjj250jjj205jj 在一个 30MB 的数组里循环读写，对齐 vs 不对齐，发现不对齐的慢了 1%
// 他做的好的地方，首先是使用了 chrono 和 vtune 都测了一遍，非常严谨
// 且设置了线程亲和性，防止操作系统把正在测试的线程调度到其他 CPU 核心上，导致 1、2 级缓存失效

// 但他的测试还是有问题的，完全可以测的更准确，区别更明显
// 首先你的数组开得很大，30MB 介于二级缓存（我是 1MB）和三级缓存（我是 12*8 = 96 MB）之间
// 要测试 CPU 指令本身的延迟，最好尽可能把数据量压在 32KB 的一级缓存内
// 否则很大一部分受限于三级缓存的速度（很慢！）如果他的数组再大一点，就变成测主存的速度了
// 而且也没有 unroll，循环的条件跳转指令也会浪费不少 CPU 周期
// 这就是为什么他用了各种高端工具，最终却只测出了 1% 的微小差别，因为各种干扰已经占了大头！

// 他视频中说是为了故意突破 CPU 缓存的容量，认为这样就能让读写绕开缓存？
// 但缓存是绕不开的，即使你访问每次都 cache miss，访问完后都是需要把当前所在的整个 cacheline
// 存起来的，因此 CPU 和内存之间通信依然是 64 字节为单位，并不是 cache 一直 miss 就可以阻止缓存
// 除非你能修改 cr0 寄存器，把 NC 设为 1，WB 设为 0，用户态可做不到这一点

// 他视频中用的是 Debug 模式，没有开启优化，可能是为了阻止矢量化，我们用 -fno-tree-vectorize
// 来模拟相同的效果，这样就不必关闭优化了

// 他在视频中提到 win 系统不支持关闭 CPU scaling，小彭老师通过：
// for x in /sys/devices/system/cpu/cpufreq/*/scaling_governor;do sudo sh -c "echo performance>$x";done
// 关闭了，希望得到更稳定的结果！

// 在下面这个案例中，Debug 模式我测得了 2% 的差距，Release 模式我测得了 10% 的差距
// Release 加上 unroll 后，得到了 14% 的差距

[[gnu::optimize("-fno-tree-vectorize")]] void BM_bigarray(benchmark::State &state) noexcept {
    size_t offset = state.range(0);
    const size_t n = 30 * 1000 * 1000; // 30 MB
    std::vector<int> arr(n + 64);
    int *p = (int *)((uintptr_t)arr.data() + offset);
    for (auto _: state) {
        #pragma GCC unroll 100
        for (size_t i = 0; i < n; i++) {
            p[i] += 1;
        }
    }
    benchmark::DoNotOptimize(arr);
}
BENCHMARK(BM_bigarray)->MinWarmUpTime(0.1)->MinTime(0.6)->ArgName("offset")->Arg(0)->Arg(1);

// 最理想的测试条件是在同一个变量上，只有一条指令，反复读取或写入，这样效果才最明显
// 此处介绍一种常见的小技巧：mov rax, [rax]
// 需要在 rax 指向的地方存一个指向自己的指针，然后不断循环这个指令
// 为什么要自己依赖自己？这样能阻止 CPU 任何指令级并行的可能性！
// 例如：
// add [rax], rcx
// add [rax + 4], rcx
// add [rax + 8], rcx
// 那么这三条指令是互相独立的！现代 CPU 都具有指令级并行的能力，可以并行执行他们
// 从而把延迟隐藏起来，这提升了效率，但会妨碍我们测出每条准确的耗时，容易受随机扰动影响
// jjj250jjj205jj 在他的测试代码中直接分配了一个 30MB 的数组，其中的每一条访问指令都互相独立
// 这会让 CPU 有机会并行，从而使结果显得不明显（只有 1%）
// 但是：
// mov rax, [rax]
// mov rax, [rax]
// mov rax, [rax]
// 就像拆链表一样，CPU 读出 rax 前，并不知道 [rax] 指向的就是自己本身
// 这样每一条指令都依赖上一条指令执行完毕，使 CPU 无法自动并行
// 这样才能无干扰地测出准确的指令耗时

// 小彭老师更加专业的测试，测出了 25% 的差距，并通过 matplotlib 可视化，最终证明
// 他的测试中不对齐会变慢的直接原因，并不是因为不对齐，而是因为有一部分访问刚好跨越了缓存行边界
// 在他的数组测试中，实际上只有 1/4 的 __m128 跨越了缓存行，只是这一部分的 __m128 降低了总体速度
// 不跨越缓存行的那些 __m128，虽然是不对齐，但实际上并没有变慢！
// 而小彭老师的测试中，访问一个 64 字节的 uintptr_t，发现只有跨越缓存行的那些访问变慢了 25%
// 没有跨越缓存行的那些，哪怕是不对齐的，速度也是完全一样！（1% 的区别都没有）
// 在一个足够长的数组中，一旦有一个对象不对齐到 2 的幂，就会有一个对象跨越缓存行
// 而他只测试了数组的情况，产生了是对齐导致减速的假象
// 实际上是对齐 + 数组，导致了必然有一个对象跨越缓存行，是那个对象拖慢了整体的速度
// 但又只有 1/4 的 __m128 跨越了，同时因为数组开得很大，三级缓存的瓶颈掩盖了太多
// 一系列的干扰项，导致最终只剩 1% 的区别，看不到真正导致变慢的原因

void BM_latency(benchmark::State &state) noexcept {
    size_t offset = state.range(0);
    const size_t repeats = 100;
    alignas(64) static char buf[256];
    uintptr_t &tmp = *new (buf + offset) uintptr_t;
    tmp = (uintptr_t)&tmp;
    uintptr_t rax = tmp;
    for (auto _: state) {
        #pragma GCC unroll repeats
        for (size_t i = 0; i < repeats; ++i) {
            rax = *(uintptr_t *)rax; // mov rax, [rax]
        }
    }
    benchmark::DoNotOptimize(rax);
}
BENCHMARK(BM_latency)->MinTime(0.05)->ArgName("offset")->DenseRange(0, 80);

// 更进一步，我们定义了一个对齐到 8 字节的结构体，但是访问时只通过 dword 访问
// 发现当 offset 为 0, 1, 2, 3, ..., 53, 54, 55, 56, 60, 64 时，均没有性能损失
// 当 offset 为 57, 58, 59, 61, 62, 63 时，才产生了性能损失
// 进一步证明，数据访问跨越了缓存行，才是性能损失的充要条件
// 如果没跨缓存行，即使访问不对齐，也不会有任何性能区别

struct alignas(8) A {
    uint32_t a;
    uint32_t b;
};

void BM_struct(benchmark::State &state) {
    size_t offset = state.range(0);
    const size_t repeats = 100;
    alignas(64) static char buf[256];
    A &tmp = *new (buf + offset) A;
    tmp.a = 0;
    tmp.b = 0;
    benchmark::DoNotOptimize(tmp);
    uintptr_t rax = (uintptr_t)&tmp;
    for (auto _: state) {
        #pragma GCC unroll repeats
        for (size_t i = 0; i < repeats; ++i) {
            // movzx rbx, dword [rax]; add rax, rbx; movzx rbx, dword [rax + 4]; add rax, rbx
            rax += ((A *)rax)->a;
            rax += ((A *)rax)->b;
        }
    }
    benchmark::DoNotOptimize(rax);
}
BENCHMARK(BM_struct)->MinTime(0.05)->ArgName("offset")->DenseRange(0, 80);

// 亚里士多德：因为我扔出一个铁球和一片羽毛，羽毛落地比铁球慢。结论：越轻的物体落地越慢
// 亚里士多德没有排除空气阻力的干扰，误以为是羽毛的“轻”属性导致落地变慢
// 实际上是羽毛的“面积-质量比”使得羽毛很容易被空气减速
// 伽利略：思想实验 + 更准确的实验 = 为什么之前的不对？排除干扰项，发现真正原因

// 不过，jjj250jjj205jj 至少确实做了测试，哪怕测试的方法有点小问题，哪怕一叶障目没找到真正的本质原因
// 也比某些心理优化、脑测仙人陈硕强多了（他在 muduo 中想当然地认为除法比三目判断快）
// 万事开头难，他敢于质疑，能手动去实测，已经是很大的进步，我们应该为 jjj250jjj205jj 鼓掌
