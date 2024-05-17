#pragma once

#include <unordered_map>
#include <vector>
#include <utility>

template <class K, class V>
struct OrderedMap {
    std::unordered_map<K, std::size_t> unomap;
    std::vector<std::pair<K, V>> order;

    using value_type = typename decltype(order)::value_type;
    using iterator = typename decltype(order)::iterator;
    using const_iterator = typename decltype(order)::const_iterator;

    bool back_swap_erase(K const &k) noexcept {
        if (auto it = unomap.find(k); it != unomap.end()) {
            std::size_t id = it->second;
            order[id] = std::move(order.back());
            order.pop_back();
            return true;
        }
        return false;
    }

    bool erase(K const &k) noexcept {
        if (auto it = unomap.find(k); it != unomap.end()) {
            std::size_t id = it->second;
            order.erase(order.begin() + id);
            return true;
        }
        return false;
    }

    iterator back_swap_erase(const_iterator it) noexcept {
        unomap.erase(it->first);
        std::size_t id = it - order.begin();
        order[id] = std::move(order.back());
        order.pop_back();
        return begin() + (it - cbegin());
    }

    iterator erase(const_iterator it) noexcept {
        unomap.erase(it->first);
        order.erase(it);
        return begin() + (it - cbegin());
    }

    std::pair<std::size_t, bool> insert(std::pair<K, V> kv) {
        return insert(std::move(kv.first), std::move(kv.second));
    }

    std::pair<std::size_t, bool> insert(K k, V v) {
        std::size_t id = order.size();
        auto [it, success] = unomap.emplace(std::as_const(k), id);
        if (success) {
            order.emplace_back(std::move(k), std::move(v));
            return {id, true};
        } else {
            return {it->second, false};
        }
    }

    std::size_t insert_or_assign(K k, V v) {
        std::size_t id = order.size();
        auto [it, success] = unomap.emplace(std::as_const(k), id);
        if (success) {
            order.emplace_back(std::move(k), std::move(v));
            return id;
        } else {
            order[it->second] = std::move(v);
        }
    }

    bool contains(K const &k) const noexcept {
        return unomap.find(k) != unomap.end();
    }

    auto find(K const &k) const noexcept {
        auto it = unomap.find(k);
        if (it == unomap.end()) {
            return order.end();
        } else {
            return order.begin() + it->second;
        }
    }

    auto find(K const &k) noexcept {
        auto it = unomap.find(k);
        if (it == unomap.end()) {
            return order.end();
        } else {
            return order.begin() + it->second;
        }
    }

    auto &at(K const &k) noexcept {
        return order[unomap.at(k)].second;
    }

    auto const &at(K const &k) const noexcept {
        return order[unomap.at(k)].second;
    }

    auto &operator[](std::size_t id) noexcept {
        return order[id].second;
    }

    auto const &operator[](std::size_t id) const noexcept {
        return order[id].second;
    }

    auto begin() noexcept {
        return order.begin();
    }

    auto end() noexcept {
        return order.end();
    }

    auto begin() const noexcept {
        return order.begin();
    }

    auto end() const noexcept {
        return order.end();
    }

    auto cbegin() const noexcept {
        return order.cbegin();
    }

    auto cend() const noexcept {
        return order.cend();
    }
};
