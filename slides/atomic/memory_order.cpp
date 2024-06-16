#include "mtpool.hpp"

struct TestNaive {
    int data = 0;
    char space[64];
    int ready = 0;

    void entry(MTIndex<0>) {
        data = 42;
        ready = 1;
    }

    void entry(MTIndex<1>) {
        while (ready == 0)
            ;
        MTTest::result = data;
    }
};

struct TestVolatile {
    int data = 0;
    char space[64];
    volatile int ready = 0;

    void entry(MTIndex<0>) {
        data = 42;
        ready = 1;
    }

    void entry(MTIndex<1>) {
        while (ready == 0)
            ;
        MTTest::result = data;
    }
};

struct TestRelaxed {
    int data = 0;
    char space[64];
    std::atomic_int ready{0};

    void entry(MTIndex<0>) {
        data = 42;
        ready.store(1, std::memory_order::relaxed);
    }

    void entry(MTIndex<1>) {
        while (ready.load(std::memory_order::relaxed) == 0)
            ;
        MTTest::result = data;
    }
};

struct TestAcquireRelease {
    int data = 0;
    char space[64];
    std::atomic_int ready{0};

    void entry(MTIndex<0>) {
        data = 42;
        // store barrier
        ready.store(1, std::memory_order::release);
    }

    void entry(MTIndex<1>) {
        while (ready.load(std::memory_order::acquire) == 0)
            ;
        // load barrier
        MTTest::result = data;
    }
};

struct TestSeqCst {
    int data = 0;
    char space[64];
    std::atomic_int ready{0};

    void entry(MTIndex<0>) {
        data = 42;
        // barrier
        ready.store(1, std::memory_order::seq_cst);
        // barrier
    }

    void entry(MTIndex<1>) {
        // barrier
        while (ready.load(std::memory_order::seq_cst) == 0)
            ;
        // barrier
        MTTest::result = data;
    }
};


int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestVolatile>();
    MTTest::runTest<TestRelaxed>();
    MTTest::runTest<TestAcquireRelease>();
    MTTest::runTest<TestSeqCst>();
    return 0;
}
