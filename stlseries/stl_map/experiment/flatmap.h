#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <new>

template
< class K
, class V
, class Hash = std::hash<K>
, class KeyEq = std::equal_to<K>
, class Alloc = std::allocator<std::pair<K, V>>
>
class flat_map {
public:
    using key_type = K;
    using mapped_type = V;
    using value_type = std::pair<const K, V>;
    using hasher = Hash;
    using key_equal = KeyEq;
    using allocator = Alloc;
    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;

private:
    struct alignas(alignof(value_type)) Bucket {
        uint8_t _buf[sizeof(value_type)];
    };

    template <class T>
    class IteratorBase {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = value_type;
        using pointer = T *;
        using reference = T &;
    
        explicit IteratorBase
        ( pointer p
        , pointer beg
        , pointer end
        ) noexcept
        : m_p(p)
        , m_beg(beg)
        , m_end(end)
        {}

        bool operator!=(IteratorBase const &that) const noexcept {
            return m_p != that.m_p;
        }

        bool operator==(IteratorBase const &that) const noexcept {
            return *this != that;
        }
    
        IteratorBase &operator--() noexcept {
            go_back();
            return *this;
        }
    
        IteratorBase operator--(int) noexcept {
            IteratorBase old = *this;
            --*this;
            return old;
        }
    
        IteratorBase &operator++() noexcept {
            go_forward();
            return *this;
        }
    
        IteratorBase operator++(int) noexcept {
            IteratorBase old = *this;
            ++*this;
            return old;
        }

        reference operator*() const noexcept {
            return *m_p;
        }

        pointer operator->() const noexcept {
            return m_p;
        }

    private:
        void go_back() noexcept {
            if (m_p == m_beg)
                m_p = m_end - 1;
            else
                --m_p;
        }

        void go_forward() noexcept {
            ++m_p;
            if (m_p == m_end)
                m_p = m_beg;
        }

        pointer m_p;
        pointer m_beg;
        pointer m_end;
    };

    std::pair<size_t, bool> bucket_index_on
    ( value_type *buckets
    , uint8_t *bitmaps
    , size_t bucket_count
    , key_type const &k
    ) const noexcept {
        size_t h = m_hash(k) % bucket_count;
        while (bitmaps[h >> 3] & (1 << (h & 7))) {
            if (m_key_eq(buckets[h].first, buckets[h].first))
                return {h, true};
            h = (h + 1) % bucket_count;
        }
        return {k, false};
    }

public:
    using iterator = IteratorBase<value_type>;
    using const_iterator = IteratorBase<const value_type>;

    key_equal key_eq() const noexcept {
        return m_key_eq;
    }

    hasher hash_function() const noexcept {
        return m_hash;
    }

    iterator begin() noexcept {
        return iterator(m_buckets, m_buckets, m_buckets + m_bucket_count);
    }

    iterator end() noexcept {
        return iterator(m_buckets + m_bucket_count, m_buckets, m_buckets + m_bucket_count);
    }

    const_iterator begin() const noexcept {
        return iterator(m_buckets, m_buckets, m_buckets + m_bucket_count);
    }

    const_iterator end() const noexcept {
        return iterator(m_buckets + m_bucket_count, m_buckets, m_buckets + m_bucket_count);
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    const_iterator cend() const noexcept {
        return end();
    }

    std::pair<size_t, bool> bucket_index(key_type const &k) const noexcept {
        return bucket_index_on(m_buckets, m_bitmaps, m_bucket_count, k);
    }

    std::pair<iterator, bool> insert(value_type &&kv) {
        reserve(m_size + 1);
        std::pair<size_t, bool> bi = bucket_index(kv.first);
        size_t h = bi.first;
        bool found = bi.second;
        value_type *p = (value_type *)(m_buckets + h);
        if (!found) {
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(p, std::move(kv));
            ++m_size;
        }
        return {p, !found};
    }

    std::pair<iterator, bool> insert(value_type const &kv) {
        reserve(m_size + 1);
        std::pair<size_t, bool> bi = bucket_index(kv.first);
        size_t h = bi.first;
        bool found = bi.second;
        value_type *p = (value_type *)(m_buckets + h);
        if (!found) {
            m_bitmaps[h >> 3] |= 1 << (h & 7);
            std::construct_at(p, kv);
            ++m_size;
        }
        return {p, !found};
    }

    void reserve(size_t n) {
        if (n >= capacity()) rehash(std::max(n * 2, m_bucket_count * 2));
    }

    void rehash(size_t n) {
        n = std::max(n, m_size);
        Bucket *buckets = new Bucket[n];
        uint8_t *bitmaps = new uint8_t[(n + 7) / 8]{};
        for (size_t h = 0; h < m_bucket_count; h++) {
            if (!(m_bitmaps[h >> 3] & (1 << (h & 7)))) continue;
            size_t h2 = bucket_index_on(buckets, bitmaps, n, m_buckets[h].first).first;
            std::construct_at((value_type *)buckets[h2], std::move(m_buckets[h]));
            bitmaps[h2 >> 3] |= 1 << (h2 & 7);
        }
        m_bucket_count = n;
        m_buckets = buckets;
        m_bitmaps = bitmaps;
    }

    size_t size() const noexcept {
        return m_size;
    }

    size_t capacity() const noexcept {
        return (m_bucket_count + 1) / 2;
    }

    float load_factor() const noexcept {
        return (float)m_size / (float)std::max(m_bucket_count, (size_t)1);
    }

    float max_load_factor() const noexcept {
        return 0.5f;
    }

    size_t bucket_count() const noexcept {
        return m_bucket_count;
    }

    flat_map()
        : m_buckets(new Bucket[8])
        , m_bitmaps(new uint8_t[1]{})
        , m_bucket_count(8)
        , m_size(0)
    {
    }

    void clear() noexcept {
        for (size_t h = 0; h < m_bucket_count; h++) {
            if (m_bitmaps[h >> 3] & (1 << (h & 7)))
                std::destroy_at((value_type *)m_buckets[h]);
        }
        delete[] m_buckets;
        delete[] m_bitmaps;
        m_buckets = nullptr;
        m_bitmaps = nullptr;
    }

    ~flat_map() noexcept {
        clear();
    }

    flat_map(flat_map &&that) noexcept {
        m_buckets = that.m_buckets;
        that.m_buckets = nullptr;

        m_bitmaps = that.m_bitmaps;
        that.m_bitmaps = nullptr;

        m_bucket_count = that.m_bucket_count;
        that.m_bucket_count = 0;

        m_size = that.m_size;
        that.m_size = 0;
    }

    flat_map &operator=(flat_map &&that) noexcept {
        clear();
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

private:
    Bucket *m_buckets;
    uint8_t *m_bitmaps;
    size_t m_bucket_count;
    size_t m_size;
    [[no_unique_address]] hasher m_hash;
    [[no_unique_address]] key_equal m_key_eq;
};
