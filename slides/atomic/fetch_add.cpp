#include "mtpool.hpp"

struct TestNaive {
    int data = 0;

    void entry(MTIndex<0>) {
        data += 1;
    }

    void entry(MTIndex<1>) {
        data += 1;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestVolatile {
    volatile int data = 0;

    void entry(MTIndex<0>) {
        data += 1;
    }

    void entry(MTIndex<1>) {
        data += 1;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomic {
    std::atomic<int> data = 0;

    void entry(MTIndex<0>) {
        data += 1;
    }

    void entry(MTIndex<1>) {
        data += 1;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomicWrong {
    std::atomic<int> data = 0;

    void entry(MTIndex<0>) {
        data = data + 1;
    }

    void entry(MTIndex<1>) {
        data = data + 1;
    }

    void teardown() {
        MTTest::result = data;
    }
};


int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestVolatile>();
    MTTest::runTest<TestAtomic>();
    MTTest::runTest<TestAtomicWrong>();
    return 0;
}
