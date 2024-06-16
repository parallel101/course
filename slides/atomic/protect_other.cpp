#include "mtpool.hpp"
using namespace std;

// atomic 有两个功能：原子 & 内存序
// 原子的作用：保证对一个变量操作期间，其他线程不会对此变量操作，那就称为原子操作
// 内存序的作用：构造一个事件发生先后的顺序关系

// 大多数普通的变量都不用是原子的，只有一个变量被多个线程同时访问（且其中一个是有副作用的写访问）时，才需要把它变成原子变量
// 所有其他变量，如果也设为原子的话，CPU 的压力会非常大，必须时时刻刻提心吊胆，编译器也会难以优化
// 我们只需要把少数多个线程同时访问的变量设为原子的就可以，帮 CPU 和编译器明确哪些是需要保证多线程并发访问安全的

//  x86 TSO
// CPU，缓存，编译器优化 -> 导致乱序
// 对少数关键变量的内存访问，指定内存序，可以阻断优化，避免这些关键变量的乱序
// 而普通的无关紧要的变量的访问依然可以乱序执行

// seq_cst = Sequentially-Consistent 顺序一致模型，总是保证所有内存操作，在所有线程看起来都是同一个顺序
// acquire = 我命令你，如果另一个线程对该变量 release 过，请把他的修改对我当前线程可见
// release = 我命令你，之前当前线程的所有写入操作，必须对所有对该变量 acquire 的线程可见
// acq_rel = acquire + release
// relaxed = 无内存序，仅保证原子性

// ARM, RISC-V

struct TestRelaxed {
    int data = 0;
    char space[64];
    std::atomic_int flag = 0;

    void entry(MTIndex<0>) {
        data = 42; // 1~5 str [data]
        flag.store(1, std::memory_order_relaxed); // 1 str [data]
    }

    void entry(MTIndex<1>) {
        while (flag.load(std::memory_order_relaxed) == 0) // 2 ldr [data]
            ;
        MTTest::result = data; // 1 ldr [data]
    }
};

struct TestSeqCst {
    int data = 0;
    char space[64];
    std::atomic_int flag = 0;

    void entry(MTIndex<0>) {
        data = 42; // 1 str [data]
        // barrier
        flag.store(1, std::memory_order_seq_cst); // 2 stlr [data]
        // barrier
    }

    void entry(MTIndex<1>) {
        // barrier
        while (flag.load(std::memory_order_seq_cst) == 0) // 3 ldar [data]
            ;
            // 可以观测到 data = 42 的副作用了
        // barrier
        MTTest::result = data; // 4 ldr [data]
    }
};

struct TestAcquireRelease {
    int data = 0;
    char space[64];
    std::atomic_int flag = 0;

    void entry(MTIndex<0>) {
        data = 42; // 1 str [data]
        // barrier v
        flag.store(1, std::memory_order_release); // 2 stlr [data]
    }

    void entry(MTIndex<1>) {
        while (flag.load(std::memory_order_acquire) == 0) // 3 ldar [data]
            ;
            // 可以观测到 data = 42 的副作用了
        // barrier ^
        MTTest::result = data; // 4 ldr [data]
    }
};

struct TestAcquireRelaxed {
    int data = 0;
    char space[64];
    std::atomic_int flag = 0;

    void entry(MTIndex<0>) {
        data = 42; // 1 str [data]
        flag.store(1, std::memory_order_relaxed); // 2 str [data]
    }

    void entry(MTIndex<1>) {
        while (flag.load(std::memory_order_acquire) == 0) // 3 ldar [data]
            ;
        // barrier ^
        MTTest::result = data; // 4 ldr [data]
    }
};

struct TestRelaxedRelease {
    int data = 0;
    char space[64];
    std::atomic_int flag = 0;

    void entry(MTIndex<0>) {
        data = 42; // 1 str [data]
        // barrier v
        flag.store(1, std::memory_order_release); // 2 stlr [data]
    }

    void entry(MTIndex<1>) {
        while (flag.load(std::memory_order_relaxed) == 0) // 3 ldr [data]
            ;
        MTTest::result = data; // 4 ldr [data]
    }
};

int main() {
    MTTest::runTest<TestRelaxed>();
    MTTest::runTest<TestSeqCst>();
    MTTest::runTest<TestAcquireRelease>();
    MTTest::runTest<TestAcquireRelaxed>();
    MTTest::runTest<TestRelaxedRelease>();
    return 0;
}
