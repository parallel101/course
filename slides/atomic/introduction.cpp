#include "mtpool.hpp"
using namespace std;

// atomic 有两个功能：原子 & 内存序
// 原子的作用：保证对一个变量操作期间，其他线程不会对此变量操作，那就称为原子操作
// 内存序的作用：构造一个事件发生先后的顺序关系

// 大多数普通的变量都不用是原子的，只有一个变量被多个线程同时访问（且其中一个是有副作用的写访问）时，才需要把它变成原子变量
// 所有其他变量，如果也设为原子的话，CPU 的压力会非常大，必须时时刻刻提心吊胆，编译器也会难以优化
// 我们只需要把少数多个线程同时访问的变量设为原子的就可以，帮 CPU 和编译器明确哪些是需要保证多线程并发访问安全的

// CPU，缓存，编译器优化 -> 导致乱序
// 对少数关键变量的内存访问，指定内存序，可以阻断优化，避免这些关键变量的乱序
// 而普通的无关紧要的变量的访问依然可以乱序执行

// seq_cst = Sequentially-Consistent 顺序一致模型
// acq_rel = acquire + release

struct Test {
    std::atomic_int data = 0;

    void entry(MTIndex<0>) {  // 0 号线程
        data.store(42, std::memory_order_relaxed);
    }

    void entry(MTIndex<1>) {  // 1 号线程
        while (data.load(std::memory_order_relaxed) == 0)
            ;
        MTTest::result = data;
    }
};
