#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <concepts>
#include "hash.h"

namespace _flat_map_details {

template
< class K
, class V
, class Hash = generic_hash<K>
, class KeyEq = std::equal_to<K>
, class Alloc = std::allocator<std::pair<const K, V>>
>
class flat_map {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const K, V>;
    using hasher = Hash;
    using key_equal = KeyEq;
    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    using allocator_type = Alloc;

private:
    using AllocTrait = std::allocator_traits<allocator_type>;
    using AllocU8Type = typename AllocTrait::template rebind_alloc<uint8_t>;
    using AllocU8Trait = std::allocator_traits<AllocU8Type>;

    template <class T>
    class IteratorBase {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = std::pair<const K, V>;
        using pointer = T *;
        using reference = T &;
    
        constexpr explicit IteratorBase
        ( pointer p
        , pointer beg
        , pointer end
        ) noexcept
        : m_p(p)
        , m_beg(beg)
        , m_end(end)
        {}

        constexpr bool operator!=(IteratorBase const &that) const noexcept {
            return m_p != that.m_p;
        }

        constexpr bool operator==(IteratorBase const &that) const noexcept {
            return *this != that;
        }
    
        constexpr IteratorBase &operator--() noexcept {
            go_back();
            return *this;
        }
    
        constexpr IteratorBase operator--(int) noexcept {
            IteratorBase old = *this;
            --*this;
            return old;
        }
    
        constexpr IteratorBase &operator++() noexcept {
            go_forward();
            return *this;
        }
    
        constexpr IteratorBase operator++(int) noexcept {
            IteratorBase old = *this;
            ++*this;
            return old;
        }

        constexpr reference operator*() const noexcept {
            return *m_p;
        }

        constexpr pointer operator->() const noexcept {
            return m_p;
        }

    private:
        constexpr void go_back() noexcept {
            if (m_p == m_beg)
                m_p = m_end - 1;
            else
                --m_p;
        }

        constexpr void go_forward() noexcept {
            ++m_p;
            if (m_p == m_end)
                m_p = m_beg;
        }

        pointer m_p;
        pointer m_beg;
        pointer m_end;

        friend flat_map;
    };

    template <class K2>
    constexpr std::pair<size_t, bool> bucket_index_on
    ( value_type *buckets
    , uint8_t *bitmaps
    , size_t bucket_count
    , K2 const &k
    ) const noexcept {
        if (!bucket_count) {
            return {(size_t)-1, false};
        }
        size_t h = m_hash(k) % bucket_count;
        while (bitmaps[h >> 3] & (1 << (h & 7))) {
            if (m_key_eq(k, buckets[h].first))
                return {h, true};
            h = (h + 1) % bucket_count;
        }
        return {h, false};
    }

public:
    using iterator = IteratorBase<value_type>;
    using const_iterator = IteratorBase<const value_type>;

    constexpr allocator_type get_allocator() const noexcept {
        return m_alloc;
    }

    constexpr key_equal key_eq() const noexcept {
        return m_key_eq;
    }

    constexpr hasher hash_function() const noexcept {
        return m_hash;
    }

    constexpr iterator begin() noexcept {
        return iterator(m_buckets, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr iterator end() noexcept {
        return iterator(m_buckets + m_bucket_count, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr const_iterator begin() const noexcept {
        return iterator(m_buckets, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr const_iterator end() const noexcept {
        return iterator(m_buckets + m_bucket_count, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr const_iterator cbegin() const noexcept {
        return begin();
    }

    constexpr const_iterator cend() const noexcept {
        return end();
    }

    constexpr std::pair<size_t, bool> bucket_index(key_type const &k) const noexcept {
        return bucket_index_on(m_buckets, m_bitmaps, m_bucket_count, k);
    }

    template <class K2>
    constexpr std::pair<size_t, bool> bucket_index(K2 const &k) const noexcept {
        return bucket_index_on(m_buckets, m_bitmaps, m_bucket_count, k);
    }

    constexpr std::pair<iterator, bool> insert(value_type &&kv) {
        reserve(m_size + 1);
        std::pair<size_t, bool> bi = bucket_index(kv.first);
        size_t h = bi.first;
        bool found = bi.second;
        value_type *p = m_buckets + h;
        if (!found) {
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(p, std::move(kv));
            ++m_size;
        }
        return {iterator(p, m_buckets, m_buckets + m_bucket_count), !found};
    }

    constexpr std::pair<iterator, bool> insert(value_type const &kv) {
        reserve(m_size + 1);
        std::pair<size_t, bool> bi = bucket_index(kv.first);
        size_t h = bi.first;
        bool found = bi.second;
        value_type *p = m_buckets + h;
        if (!found) {
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(p, kv);
            ++m_size;
        }
        return {iterator(p, m_buckets, m_buckets + m_bucket_count), !found};
    }

    constexpr bool erase(key_type const &k) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (found) {
            m_bitmaps[h >> 3] &= ~(1 << (h & 7));
            std::destroy_at(m_buckets + h);
            --m_size;
            return true;
        }
        return false;
    }

    constexpr iterator erase(iterator pos) {
        size_t h = pos.m_p - m_buckets;
        m_bitmaps[h >> 3] &= ~(1 << (h & 7));
        std::destroy_at(m_buckets + h);
        --m_size;
        return pos;
    }

    constexpr bool contains(key_type const &k) const noexcept {
        return bucket_index(k).second;
    }

    template <class K2>
    constexpr bool contains(K2 const &k) const noexcept {
        return bucket_index(k).second;
    }

    constexpr iterator find(key_type const &k) noexcept {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            return end();
        }
        return iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count);
    }

    template <class K2>
    constexpr iterator find(K2 const &k) noexcept {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            return end();
        }
        return iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr const_iterator find(key_type const &k) const noexcept {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            return end();
        }
        return const_iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count);
    }

    template <class K2>
    constexpr const_iterator find(K2 const &k) const noexcept {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            return end();
        }
        return const_iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count);
    }

    constexpr mapped_type &at(key_type const &k) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        [[unlikely]] if (!found) {
            throw std::out_of_range("flat_map::at");
        }
        return m_buckets[h].second;
    }

    template <class K2>
    constexpr mapped_type &at(K2 const &k) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        [[unlikely]] if (!found) {
            throw std::out_of_range("flat_map::at");
        }
        return m_buckets[h].second;
    }

    constexpr mapped_type const &at(key_type const &k) const {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        [[unlikely]] if (!found) {
            throw std::out_of_range("flat_map::at");
        }
        return m_buckets[h].second;
    }

    template <class K2>
    constexpr mapped_type const &at(K2 const &k) const {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        [[unlikely]] if (!found) {
            throw std::out_of_range("flat_map::at");
        }
        return m_buckets[h].second;
    }

    constexpr mapped_type &operator[](key_type const &k) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            if (m_size + 1 > capacity()) {
                reserve(m_size + 1);
                bi = bucket_index(k);
                h = bi.first;
                found = bi.second;
            }
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(m_buckets + h, std::piecewise_construct, std::make_tuple(k), std::make_tuple());
            ++m_size;
        }
        return m_buckets[h].second;
    }

    template <class V2>
    constexpr std::pair<iterator, bool> insert_or_assign(key_type const &k, V2 &&v) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            if (m_size + 1 > capacity()) {
                reserve(m_size + 1);
                bi = bucket_index(k);
                h = bi.first;
                found = bi.second;
            }
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(m_buckets + h, std::piecewise_construct, std::forward_as_tuple(k), std::forward_as_tuple(v));
            ++m_size;
        }
        m_buckets[h] = std::forward<V2>(v);
        return {iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count), !found};
    }

    template <class V2>
    constexpr std::pair<iterator, bool> insert_or_assign(key_type &&k, V2 &&v) {
        std::pair<size_t, bool> bi = bucket_index(k);
        size_t h = bi.first;
        bool found = bi.second;
        if (!found) {
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(m_buckets + h, std::piecewise_construct, std::forward_as_tuple(k), std::forward_as_tuple(v));
            ++m_size;
        }
        m_buckets[h] = std::forward<V2>(v);
        return {iterator(m_buckets + h, m_buckets, m_buckets + m_bucket_count), !found};
    }

    constexpr void reserve(size_t n) {
        if (n >= capacity()) rehash(std::max(n * 2, m_bucket_count * 2));
    }

    constexpr void shrink_to_fit() {
        if (m_bucket_count > m_size * 2) rehash(0);
    }

    constexpr void rehash(size_t n) {
        n = std::max(n, m_size * 2);
        if (n) {
            value_type *buckets = AllocTrait::allocate(m_alloc, n);
            uint8_t *bitmaps = AllocU8Trait::allocate(m_alloc_u8, (n + 7) / 8);
            for (size_t i = 0; i < (n + 7) / 8; i++) bitmaps[i] = 0;
            if (m_bucket_count) {
                for (size_t h = 0; h < m_bucket_count; h++) {
                    if (!(m_bitmaps[h >> 3] & (1 << (h & 7)))) continue;
                    size_t h2 = bucket_index_on(buckets, bitmaps, n, m_buckets[h].first).first;
                    std::construct_at(buckets + h2, std::move(m_buckets[h]));
                    std::destroy_at(m_buckets + h);
                    bitmaps[h2 >> 3] |= 1 << (h2 & 7);
                }
                AllocTrait::deallocate(m_alloc, m_buckets, m_bucket_count);
                AllocU8Trait::deallocate(m_alloc_u8, m_bitmaps, (m_bucket_count + 7) / 8);
            }
            m_bucket_count = n;
            m_buckets = buckets;
            m_bitmaps = bitmaps;
        }
    }

    constexpr size_t size() const noexcept {
        return m_size;
    }

    constexpr size_t capacity() const noexcept {
        return (m_bucket_count + 1) / 2;
    }

    constexpr float load_factor() const noexcept {
        return (float)m_size / (float)std::max(m_bucket_count, (size_t)1);
    }

    constexpr float max_load_factor() const noexcept {
        return 0.5f;
    }

    constexpr size_t bucket_count() const noexcept {
        return m_bucket_count;
    }

    constexpr flat_map()
        : m_buckets(nullptr)
        , m_bitmaps(nullptr)
        , m_bucket_count(0)
        , m_size(0)
    {}

    template <std::input_iterator InputIt, std::sentinel_for<InputIt> InputSen>
    constexpr flat_map(InputIt first, InputSen last) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    template <std::input_iterator InputIt, std::sentinel_for<InputIt> InputSen>
    constexpr void insert(InputIt first, InputSen last) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    constexpr void clear() noexcept {
        m_size = 0;
        for (size_t h = 0; h < m_bucket_count; h++) {
            if (m_bitmaps[h >> 3] & (1 << (h & 7))) {
                m_bitmaps[h >> 3] &= ~(1 << (h & 7));
                std::destroy_at(m_buckets + h);
            }
        }
    }

    constexpr ~flat_map() noexcept {
        if (m_bucket_count) {
            for (size_t h = 0; h < m_bucket_count; h++) {
                if (m_bitmaps[h >> 3] & (1 << (h & 7)))
                    std::destroy_at(m_buckets + h);
            }
            AllocTrait::deallocate(m_alloc, m_buckets, m_bucket_count);
            AllocU8Trait::deallocate(m_alloc_u8, m_bitmaps, (m_bucket_count + 7) / 8);
            m_buckets = nullptr;
            m_bitmaps = nullptr;
            m_bucket_count = 0;
        }
    }

    constexpr flat_map(flat_map &&that) noexcept {
        m_buckets = that.m_buckets;
        that.m_buckets = nullptr;

        m_bitmaps = that.m_bitmaps;
        that.m_bitmaps = nullptr;

        m_bucket_count = that.m_bucket_count;
        that.m_bucket_count = 0;

        m_size = that.m_size;
        that.m_size = 0;
    }

    constexpr flat_map &operator=(flat_map &&that) noexcept {
        for (size_t h = 0; h < m_bucket_count; h++) {
            if (m_bitmaps[h >> 3] & (1 << (h & 7)))
                std::destroy_at(m_buckets + h);
        }
        AllocTrait::deallocate(m_alloc, m_buckets, m_bucket_count);
        AllocU8Trait::deallocate(m_alloc_u8, m_bitmaps, (m_bucket_count + 7) / 8);

        m_buckets = that.m_buckets;
        that.m_buckets = nullptr;
        m_bitmaps = that.m_bitmaps;
        that.m_bitmaps = nullptr;
        m_bucket_count = that.m_bucket_count;
        that.m_bucket_count = 0;
        m_size = that.m_size;
        that.m_size = 0;
        return *this;
    }

    constexpr value_type *bucket_data() {
        return m_buckets;
    }

    constexpr value_type const *bucket_data() const {
        return m_buckets;
    }

private:
    value_type *m_buckets;
    uint8_t *m_bitmaps;
    size_t m_bucket_count;
    size_t m_size;
    [[no_unique_address]] hasher m_hash;
    [[no_unique_address]] key_equal m_key_eq;
    [[no_unique_address]] allocator_type m_alloc;
    [[no_unique_address]] AllocU8Type m_alloc_u8;
};

}

using _flat_map_details::flat_map;
