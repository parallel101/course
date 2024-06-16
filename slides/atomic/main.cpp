#include "mtpool.hpp"
#include <barrier>

struct TestNoBarrier {
    int data0;
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
    }

    void entry(MTIndex<1>) {
        data1 = 24;
    }

    void entry(MTIndex<2>) {
        MTTest::result = data0 + data1;
    }
};

struct TestStdBarrier {
    std::barrier<> b{3};
    int data0;
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        b.arrive_and_wait();
        // barrier
        MTTest::result = data0 + data1;
    }
};

struct TestAtomicSeqCst {
    std::atomic<int> count{0};
    char space0[64];
    int data0;
    char space1[64];
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        count++;
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        count++;
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        while (count != 2);
        // barrier
        MTTest::result = data0 + data1;
    }
};

struct TestAtomicRelaxed {
    std::atomic<int> count{0};
    char space0[64];
    int data0;
    char space1[64];
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        count.fetch_add(1, std::memory_order_relaxed);
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        count.fetch_add(1, std::memory_order_relaxed);
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        while (count.load(std::memory_order_relaxed) != 2);
        // barrier
        MTTest::result = data0 + data1;
    }
};

struct TestAtomicAcqRel {
    std::atomic<int> count{0};
    char space0[64];
    int data0;
    char space1[64];
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        count.fetch_add(1, std::memory_order_release);
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        count.fetch_add(1, std::memory_order_release);
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        while (count.load(std::memory_order_acquire) != 2);
        // barrier
        MTTest::result = data0 + data1;
    }
};

int fetch_sub(std::atomic<int> counter, int delta) {
    int old_counter = counter;
    counter = old_counter - delta;
    return old_counter;
}

struct MySpinBarrier {
    std::atomic<int> counter;
    int const workers;
    std::atomic<int> step;

    explicit MySpinBarrier(int workers_) : counter(workers_), workers(workers_) {
    }

    void arrive_and_wait() {
        int oldstep = step.load(std::memory_order_relaxed);
        int old_counter = counter.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (old_counter == 0) {
            counter.store(workers, std::memory_order_relaxed);
            step.fetch_add(1, std::memory_order_release);
#if __cpp_lib_atomic_wait
            step.notify_all();
#endif
        } else {
            while (step.load(std::memory_order_acquire) == oldstep) {
#if __cpp_lib_atomic_wait
                step.wait(oldstep, std::memory_order_relaxed);
#endif
            }
        }
    }
};

struct TestSpinBarrier {
    MySpinBarrier b{3};
    int data0;
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        b.arrive_and_wait();
        // barrier
        MTTest::result = data0 + data1;
    }
};

struct TestSpinBarrierMulti {
    MySpinBarrier b{3};
    int data0;
    int data1;

    void entry(MTIndex<0>) {
        data0 = 42;
        // barrier
        b.arrive_and_wait();
        b.arrive_and_wait();
        // barrier
        data0 = 42;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<1>) {
        data1 = 24;
        // barrier
        b.arrive_and_wait();
        b.arrive_and_wait();
        // barrier
        data1 = 24;
        // barrier
        b.arrive_and_wait();
        // barrier
    }

    void entry(MTIndex<2>) {
        // barrier
        b.arrive_and_wait();
        // barrier
        MTTest::result = data0 + data1;
        // barrier
        b.arrive_and_wait();
        b.arrive_and_wait();
        // barrier
        MTTest::result = data0 + data1;
    }
};

int main() {
    MTTest::runTest<TestNoBarrier>();
    MTTest::runTest<TestStdBarrier>();
    MTTest::runTest<TestAtomicSeqCst>();
    MTTest::runTest<TestAtomicRelaxed>();
    MTTest::runTest<TestAtomicAcqRel>();
    MTTest::runTest<TestSpinBarrier>();
    MTTest::runTest<TestSpinBarrierMulti>();
    return 0;
}
