#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"

struct Node {
    Node *next;
    int value;

    explicit Node(int value_) {
        next = nullptr;
        value = value_;
    }
};

struct ConcurrentList {
    struct TagNodePtr {
#if __x86_64__ || __arm64__
        uintptr_t node_raw: 48;
        uint16_t tag: 16;
#else
        uintptr_t node_raw;
        uint64_t tag;
#endif

        Node *node() const {
            return reinterpret_cast<Node *>(node_raw);
        }

        static TagNodePtr make(Node *node_, uint16_t tag_) {
            return {reinterpret_cast<uintptr_t>(node_), tag_};
        }
    };

    std::atomic<TagNodePtr> head = TagNodePtr::make(nullptr, 0);

    void push_front(Node *node) {
        TagNodePtr oldhead = head.load(std::memory_order_acquire);
        TagNodePtr newhead;
        do {
            node->next = oldhead.node();
            newhead = TagNodePtr::make(node, oldhead.tag + 1);
        } while (!head.compare_exchange_weak(oldhead, newhead, std::memory_order_acq_rel, std::memory_order_relaxed));
    }

    Node *pop_front() {
        TagNodePtr oldhead = head.load(std::memory_order_acquire);
        TagNodePtr newhead;
        do {
            if (oldhead.node() == nullptr)
                return nullptr;
            newhead = TagNodePtr::make(oldhead.node()->next, oldhead.tag + 1);
        } while (!head.compare_exchange_weak(oldhead, newhead, std::memory_order_acq_rel, std::memory_order_relaxed));
        return oldhead.node();
    }
};

ConcurrentList a;

void t1() {
    auto p = a.pop_front();
    if (p)
        a.push_front(p);
}

void t2() {
    auto p = a.pop_front();
    if (p)
        a.push_front(p);
}

void t3() {
    auto p = a.pop_front();
    if (p)
        a.push_front(p);
}

int main() {
    a.push_front(new Node(1));
    a.push_front(new Node(2));
    a.push_front(new Node(3));
    a.push_front(new Node(4));
    ParallelPool pool{t1, t2, t3};
    pool.join();
    for (Node *temp = a.head.load().node(); temp != nullptr; temp = temp->next) {
        std::cout << temp->value << ' ';
    }
    return 0;
}
