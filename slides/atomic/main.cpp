#include "mtpool.hpp"

struct TestNaive {
    int data = 0;

    void entry(MTIndex<0>) {
        data++;
    }

    void entry(MTIndex<1>) {
        data++;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomic {
    std::atomic<int> data = 0;

    void entry(MTIndex<0>) {
        data++;
    }

    void entry(MTIndex<1>) {
        data++;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomicRelaxed {
    std::atomic<int> data = 0;

    void entry(MTIndex<0>) {
        data.fetch_add(1, std::memory_order::relaxed);
    }

    void entry(MTIndex<1>) {
        data.fetch_add(1, std::memory_order::relaxed);
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
    MTTest::runTest<TestAtomic>();
    MTTest::runTest<TestAtomicRelaxed>();
    MTTest::runTest<TestAtomicWrong>();
    return 0;
}
