#include "mtpool.hpp"
#include <list>

struct TestNaive {
    std::list<int> data;

    void entry(MTIndex<0>) {
        data.push_back(1);
    }

    void entry(MTIndex<1>) {
        data.push_back(2);
    }

    void teardown() {
        MTTest::result = data.size(); // 应为 2，但实际上出错
    }
};

struct TestMutex {
    std::list<int> data;
    std::mutex mutex;

    void entry(MTIndex<0>) {
        mutex.lock();
        data.push_back(1);
        mutex.unlock();
    }

    void entry(MTIndex<1>) {
        mutex.lock();
        data.push_back(2);
        mutex.unlock();
    }

    void teardown() {
        MTTest::result = data.size(); // 应为 2
    }
};

struct MyList {
    struct Node {
        Node *next;
        int value;
    };

    std::atomic<Node *> head{nullptr};

    void push_back(int value) {
        Node *new_node = new Node;
        new_node->value = value;

        Node *old_head = head.load(std::memory_order_relaxed);
        do
            new_node->next = old_head;
        // store barrier
        while (!head.compare_exchange_weak(old_head, new_node, std::memory_order_release, std::memory_order_relaxed));
    }

    int pop_back() {
        Node *old_head = head.load(std::memory_order_relaxed);
        do {
            if (old_head == nullptr)
                return -1;
        } while (!head.compare_exchange_weak(old_head, old_head->next, std::memory_order_acquire, std::memory_order_relaxed));
        // load barrier
        int value = old_head->value;
        delete old_head;
        return value;
    }

    int size_unsafe() {
        int count = 0;
        Node *node = head.load();
        while (node != nullptr) {
            count++;
            node = node->next;
        }
        return count;
    }
};

struct TestMyList {
    MyList data;

    void entry(MTIndex<0>) {
        data.push_back(1);
    }

    void entry(MTIndex<1>) {
        data.push_back(2);
    }

    void teardown() {
        MTTest::result = data.size_unsafe(); // 应为 2
    }
};

int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestMutex>();
    MTTest::runTest<TestMyList>();
    return 0;
}
