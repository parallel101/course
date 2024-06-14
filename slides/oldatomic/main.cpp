#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"
#include <set>

struct AntiABAConcurrentList {
    struct Node {
        Node *next;
        int value;

        explicit Node(int value_) {
            next = nullptr;
            value = value_;
        }
    };

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

    std::atomic<TagNodePtr> head{TagNodePtr::make(nullptr, 0)};

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

    Node *front() {
        return head.load(std::memory_order_acquire).node();
    }
};

struct ConcurrentList {
    struct Node {
        Node *next;
        int value;

        explicit Node(int value_) {
            next = nullptr;
            value = value_;
        }
    };

    std::atomic<Node *> head{nullptr};

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

    Node *front() {
        return head.load(std::memory_order_acquire);
    }
};

struct ConcurrentUniqueList {
    struct Node {
        std::atomic<Node *> next{nullptr};
        std::atomic<Node *> prev{nullptr};
        int value;

        explicit Node(int value_) {
            value = value_;
        }
    };
    
    std::atomic<Node *> head{nullptr};
    std::atomic<Node *> tail{nullptr};

    bool insert(int i) {
        Node *newnode = new Node(i);
        while (true) {
            Node *oldtail = tail.load();
            Node *oldnext = oldtail->next.load();
            if (tail == tail.load()) {
                if (oldnext == nullptr) {
                    if (oldtail->next.compare_exchange_weak(oldnext, newnode)) {
                        tail.compare_exchange_strong(oldtail, newnode);
                        return true;
                    }
                } else {
                    tail.compare_exchange_strong(oldtail, oldnext);
                }
            }
        }
    }

    Node *front() {
        return head.load(); 
    }

    bool find(int i) {
        Node *oldhead = head.load();
        while (oldhead != nullptr) {
            if (oldhead->value == i) {
                return true;
            }
            oldhead = oldhead->next.load();
        }
        return false;
    }
};

struct ConcurrentHashTable {
    ConcurrentUniqueList hash_table[8];

    int hash(int i) {
        return (i * 17) % 8;
    }

    bool insert(int i) {
        return hash_table[hash(i)].insert(i);
    }

    bool find(int i) {
        auto *head = hash_table[hash(i)].front();
        while (head != nullptr) {
            if (head->value == i) {
                return true;
            }
            head = head->next;
        }
        return false;
    }
};

ConcurrentHashTable ht;

void t1() {
    for (int i = 0; i < 5; i++) {
        ht.insert(i);
    }
}

void t2() {
    for (int i = 5; i < 20; i++) {
        ht.insert(i);
    }
    std::cout << "INSERT " + std::to_string(ht.insert(8)) + '\n';
}

void t3() {
    while (!ht.find(10))
        ;
    std::cout << "FOUND " + std::to_string(ht.find(8)) + '\n';
}

int main() {
    ParallelPool pool{t1, t2, t3};
    pool.join();
    std::set<int> values;
    for (size_t i = 0; i < std::size(ht.hash_table); i++) {
        for (auto *temp = ht.hash_table[i].head.load(); temp != nullptr; temp = temp->next) {
            values.insert(temp->value);
        }
    }
    for (auto const &value: values) {
        std::cout << value << ' ';
    }
    return 0;
}
