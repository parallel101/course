#pragma once
#include <atomic>

template <class T>
struct ConcurrentList {
    ConcurrentList() = default;
    ConcurrentList(ConcurrentList &&) = delete;

    struct Node {
        Node *next;
        T value;
    };

    std::atomic<Node *> head{nullptr};

    void push_back(T value) {
        Node *new_node = new Node;
        new_node->value = std::move(value);

        Node *old_head = head.load(std::memory_order_relaxed);
        do
            new_node->next = old_head;
        // store barrier
        while (!head.compare_exchange_weak(old_head, new_node, std::memory_order_release, std::memory_order_relaxed));
#if __cpp_lib_atomic_wait
        head.notify_one();
#endif
    }

    bool pop_back_nowait(T &value) {
        Node *old_head = head.load(std::memory_order_consume);
        do {
            if (old_head == nullptr)
                return false;
        } while (!head.compare_exchange_weak(old_head, old_head->next, std::memory_order_consume, std::memory_order_consume));
        // load barrier
        value = std::move(old_head->value);
        delete old_head;
        return true;
    }

    T pop_back() {
        Node *old_head = head.load(std::memory_order_consume);
        do {
            while (old_head == nullptr) {
#if __cpp_lib_atomic_wait
                int retries = 200;
                if (retries <= 0) {
                    head.wait(nullptr, std::memory_order_relaxed);
                    --retries;
                }
#endif
                old_head = head.load(std::memory_order_consume);
            }
        } while (!head.compare_exchange_weak(old_head, old_head->next, std::memory_order_consume, std::memory_order_consume));
        // load barrier
        T value = std::move(old_head->value);
        delete old_head;
        return value;
    }

    size_t size() const {
        size_t count = 0;
        Node *node = head.load(std::memory_order_acquire);
        while (node != nullptr) {
            ++count;
            node = node->next;
        }
        return count;
    }
};
