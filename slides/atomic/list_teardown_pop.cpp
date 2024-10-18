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
        while (data.empty())
            ;
        MTTest::result = data.back();
        data.pop_back();
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

    void entry(MTIndex<2>) {
        mutex.lock();
        while (data.empty()) {
            mutex.unlock();
            std::this_thread::yield();
            mutex.lock();
        }
        MTTest::result = data.back();
        data.pop_back();
        mutex.unlock();
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

        Node *old_head = head.load(std::memory_order_consume);
        do
            new_node->next = old_head;
        // store barrier
        while (!head.compare_exchange_weak(old_head, new_node, std::memory_order_release, std::memory_order_consume));
    }

    int pop_back() {
        Node *old_head = head.load(std::memory_order_consume);
        do {
            if (old_head == nullptr)
                return -1;
        } while (!head.compare_exchange_weak(old_head, old_head->next, std::memory_order_acquire, std::memory_order_consume));
        // load barrier
        int value = old_head->value;
        delete old_head;
        return value;
    }

    int size() {
        int count = 0;
        Node *node = head.load(std::memory_order_acquire);
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

    void entry(MTIndex<2>) {
        int tmp;
        do
            tmp = data.pop_back();
        while (tmp == -1);
        MTTest::result = tmp;
    }
};

struct ConcurrentList {
    ConcurrentList() = default;
    ConcurrentList(ConcurrentList &&) = delete;

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
#if __cpp_lib_atomic_wait
        head.notify_one();
#endif
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

    int pop_back_wait() {
        Node *old_head = head.load(std::memory_order_relaxed);
        do {
            if (old_head == nullptr) {
#if __cpp_lib_atomic_wait
                int retries = 200;
                if (retries <= 0) {
                    head.wait(nullptr, std::memory_order_relaxed);
                    --retries;
                } else {
                    old_head = head.load(std::memory_order_relaxed);
                    continue;
                }
#else
                old_head = head.load(std::memory_order_relaxed);
                continue;
#endif
            }
        } while (!head.compare_exchange_weak(old_head, old_head->next, std::memory_order_acquire, std::memory_order_relaxed));
        // load barrier
        int value = old_head->value;
        delete old_head;
        return value;
    }

    int size() {
        int count = 0;
        Node *node = head.load(std::memory_order_acquire);
        while (node != nullptr) {
            count++;
            node = node->next;
        }
        return count;
    }
};

struct TestConcurrentList {
    ConcurrentList data;

    void entry(MTIndex<0>) {
        data.push_back(1);
    }

    void entry(MTIndex<1>) {
        data.push_back(2);
    }

    void entry(MTIndex<2>) {
        MTTest::result = data.pop_back();
    }
};

int main() {
    MTTest::runTest<TestNaive>();
    MTTest::runTest<TestMutex>();
    MTTest::runTest<TestMyList>();
    MTTest::runTest<TestConcurrentList>(1);
    return 0;
}
