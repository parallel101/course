#include "mtpool.hpp"

struct TestNaive {
    int data = 2;

    void entry(MTIndex<0>) {
        data = data * data;
    }

    void entry(MTIndex<1>) {
        data = data * data;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomicWrong {
    std::atomic<int> data = 2;

    void entry(MTIndex<0>) {
        data = data * data;
        // data.store(data.load() * data.load()); // ERR
        // data.fetch_add(1); // OK
        // data.store(data.load() + 1); // ERR
    }

    void entry(MTIndex<1>) {
        data = data * data;
    }

    void teardown() {
        MTTest::result = data;
    }
};

// CAS 指令内部原理如下
bool compare_exchange_strong(std::atomic<int> &data, int &old_data, int new_data) {
    if (data == old_data) {
        data = new_data;
        return true;
    } else {
        old_data = data;
        return false;
    }
}
bool compare_exchange_weak(std::atomic<int> &data, int &old_data, int new_data) {
    if (data == old_data && rand()) {
        data = new_data;
        return true;
    } else {
        old_data = data;
        return false;
    }
}

struct TestAtomicCAS { // compare-and-swap
    std::atomic<int> data = 2;

    void entry(MTIndex<0>) {
        auto old_data = data.load(std::memory_order_relaxed);
    again:
        auto new_data = old_data * old_data;
        if (!data.compare_exchange_strong(old_data, new_data, std::memory_order_relaxed))
            goto again;
    }

    void entry(MTIndex<1>) {
        auto old_data = data.load(std::memory_order_relaxed);
    again:
        auto new_data = old_data * old_data;
        if (!data.compare_exchange_strong(old_data, new_data, std::memory_order_relaxed))
            goto again;
    }

    void teardown() {
        MTTest::result = data;
    }
};

struct TestAtomicCASWeak { // compare-and-swap
    std::atomic<int> data = 2;

    void entry(MTIndex<0>) {
        auto old_data = data.load(std::memory_order_relaxed);
    again:
        auto new_data = old_data * old_data;
        if (!data.compare_exchange_weak(old_data, new_data, std::memory_order_relaxed))
            goto again;
    }

    void entry(MTIndex<1>) {
        auto old_data = data.load(std::memory_order_relaxed);
    again:
        auto new_data = old_data * old_data;
        if (!data.compare_exchange_weak(old_data, new_data, std::memory_order_relaxed))
            goto again;
    }

    void teardown() {
        MTTest::result = data;
    }
};

int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestAtomicWrong>();
    MTTest::runTest<TestAtomicCAS>();
    MTTest::runTest<TestAtomicCASWeak>();
    return 0;
}
