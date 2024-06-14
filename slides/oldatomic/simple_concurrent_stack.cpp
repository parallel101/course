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
    std::atomic<Node *> head = nullptr;

    void push_front(Node *node) {
        Node *oldhead = head.load();
        do {
            node->next = oldhead;
            std::this_thread::yield();
        } while (!head.compare_exchange_weak(oldhead, node));
    }

    Node *pop_front() {
        Node *oldhead = head.load();
        Node *newhead;
        do {
            if (oldhead == nullptr)
                return nullptr;
            newhead = oldhead->next;
        } while (!head.compare_exchange_weak(oldhead, newhead));
        return oldhead;
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
    for (Node *temp = a.head; temp != nullptr; temp = temp->next) {
        std::cout << temp->value << ' ';
    }
    return 0;
}
