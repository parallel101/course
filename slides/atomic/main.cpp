#include "mtpool.hpp"

struct Test0 {
    int data = 0;
    char space[64];
    int ready = 0;

    [[gnu::noinline]] void entry(MTIndex<0>) {
        data = 42;
        ready = 1;
    }

    [[gnu::noinline]] void entry(MTIndex<1>) {
        while (ready == 0)
            ;
        MTTest::result = data;
    }
};

struct Test1 {
    int data = 0;
    char space[64];
    volatile int ready = 0;

    [[gnu::noinline]] void entry(MTIndex<0>) {
        data = 42;
        ready = 1;
    }

    [[gnu::noinline]] void entry(MTIndex<1>) {
        while (ready == 0)
            ;
        MTTest::result = data;
    }
};

struct Test2 {
    int data = 0;
    char space[64];
    std::atomic_int ready{0};

    [[gnu::noinline]] void entry(MTIndex<0>) {
        data = 42;
        ready.store(1, std::memory_order::relaxed);
    }

    [[gnu::noinline]] void entry(MTIndex<1>) {
        while (ready.load(std::memory_order::relaxed) == 0)
            ;
        MTTest::result = data;
    }
};

struct Test3 {
    int data = 0;
    char space[64];
    std::atomic_int ready{0};

    [[gnu::noinline]] void entry(MTIndex<0>) {
        data = 42;
        ready.store(1, std::memory_order::release);
    }

    [[gnu::noinline]] void entry(MTIndex<1>) {
        while (ready.load(std::memory_order::acquire) == 0)
            ;
        MTTest::result = data;
    }
};


int main() {
    /* MTTest::runTest<Test1>(10); */
    MTTest::runTest<Test0>();
    MTTest::runTest<Test1>();
    MTTest::runTest<Test2>();
    MTTest::runTest<Test3>();
    return 0;
}
