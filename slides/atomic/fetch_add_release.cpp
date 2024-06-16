#include "mtpool.hpp"

struct TestNaive {
    int data = 0;
    char space[64];
    int count = 0;

    void entry(MTIndex<0>) {
        data = 42;
        count++;
    }

    void entry(MTIndex<1>) {
        while (count == 0)
            ;
        MTTest::result = data;
    }
};

struct TestAtomic {
    int data = 0;
    char space[64];
    std::atomic<int> count = 0;

    void entry(MTIndex<0>) {
        data = 42;
        // barrier
        count++;
        // barrier
    }

    void entry(MTIndex<1>) {
        // barrier
        while (count == 0)
            ;
        // barrier
        MTTest::result = data;
    }
};

struct TestAtomicRelaxed {
    int data = 0;
    char space[64];
    std::atomic<int> count = 0;

    void entry(MTIndex<0>) {
        data = 42;
        count.fetch_add(1, std::memory_order_relaxed);
    }

    void entry(MTIndex<1>) {
        while (count.load(std::memory_order_relaxed) == 0)
            ;
        MTTest::result = data;
    }
};

struct TestAtomicAcqRel {
    int data = 0;
    char space[64];
    std::atomic<int> count = 0;

    void entry(MTIndex<0>) {
        data = 42;
        // barrier
        count.fetch_add(1, std::memory_order_release);
    }

    void entry(MTIndex<1>) {
        while (count.load(std::memory_order_acquire) == 0)
            ;
        // barrier
        MTTest::result = data;
    }
};

int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestAtomic>();
    MTTest::runTest<TestAtomicRelaxed>();
    MTTest::runTest<TestAtomicAcqRel>();
    return 0;
}
