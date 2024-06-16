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

struct TestMutex {
    int data = 0;
    std::mutex mutex;

    void entry(MTIndex<0>) {
        mutex.lock();
        data += 1;
        mutex.unlock();
    }

    void entry(MTIndex<1>) {
        mutex.lock();
        data += 1;
        mutex.unlock();
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct SpinMutex {
    void lock() {
    }

    void unlock() {
    }
};

struct TestSpinMutex {
    int data = 0;
    SpinMutex mutex;

    void entry(MTIndex<0>) {
        mutex.lock();
        data += 1;
        mutex.unlock();
    }

    void entry(MTIndex<1>) {
        mutex.lock();
        data += 1;
        mutex.unlock();
    }

    void teardown() {
        MTTest::result = data;
    }
};

int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestMutex>();
    MTTest::runTest<TestSpinMutex>();
    return 0;
}
