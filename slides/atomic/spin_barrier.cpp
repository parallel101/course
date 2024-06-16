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

struct TestAtomic {
    std::atomic<int> count{0};
    int data0;
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

int main() {
    MTTest::runTest<TestNoBarrier>();
    MTTest::runTest<TestStdBarrier>();
    MTTest::runTest<TestAtomic>();
    return 0;
}
