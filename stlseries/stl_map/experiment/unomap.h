#pragma once

#include <vector>
#include <memory>

template
< class K
, class V
, class Hash = std::hash<K>
, class KeyEq = std::equal_to<K>
, class Alloc = std::allocator<std::pair<const K, V>>
>
class unordered_map {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const K, V>;
    using hasher = Hash;
    using key_equal = KeyEq;
    using allocator = Alloc;

private:
    struct Node {
        Node *next;
        value_type value;
    };

    using AllocNode = typename std::allocator_traits<allocator>::template rebind_alloc<Node>;

public:
    class iterator {
        Node *m_node;

        explicit iterator(Node *node) : m_node(node) {
        }

        bool operator!=(iterator const &that) const {
            return m_node != that.m_node;
        }

        bool operator==(iterator const &that) const {
            return *this != that;
        }

        iterator &operator++() {
            m_node = m_node->next;
            return *this;
        }

        iterator operator++(int) {
            iterator old = *this;
            ++*this;
            return old;
        }
    };

    std::pair<iterator, bool> insert(value_type kv) {
        if (m_size + 1 > m_buckets.size()) reserve(m_size + 1);
        int h = hash(kv.first) % m_buckets.size();
        for (Node *node = m_buckets[h]; node; node = node->next) {
            if (kv.first == node->value.first)
                return {iterator(node), false};
        }
        Node *new_node = new Node;
        new_node->next = m_buckets[h];
        m_buckets[h] = new_node;
        m_size++;
        return {iterator(new_node), true};
    }

    void reserve(size_t n) {
        if (n <= m_buckets.size()) return;
        m_buckets.resize(max(n, m_buckets.size() * 2));
        // rehash
    }

private:
    std::vector<Node *> m_buckets;
    size_t m_size;
};
