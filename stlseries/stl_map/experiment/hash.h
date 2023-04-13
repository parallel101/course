#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace _hash_details {

// copied from boost:
// https://www.boost.org/doc/libs/1_64_0/boost/functional/hash/hash.hpp

constexpr std::uint32_t _bit_rotate_left(std::uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

template <class SizeT>
constexpr void _hash_combine_impl(SizeT &seed, SizeT value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

constexpr std::uint32_t hash_combine_32(std::uint32_t h1, std::uint32_t k1) {
    const std::uint32_t c1 = 0xcc9e2d51;
    const std::uint32_t c2 = 0x1b873593;

    k1 *= c1;
    k1 = _bit_rotate_left(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = _bit_rotate_left(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;

    return h1;
}

constexpr std::uint64_t hash_combine_64(std::uint64_t h, std::uint64_t k) {
    const std::uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
    const int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    // Completely arbitrary number, to prevent 0's
    // from hashing to 0.
    h += 0xe6546b64;

    return h;
}

template <class T>
constexpr void hash_combine(std::size_t &seed, T const &v) {
    std::hash<T> hasher;
    return _hash_combine_impl(seed, hasher(v));
}

template <class It, class T = typename std::iterator_traits<It>::value_type>
constexpr std::size_t hash_range(It first, It last,
                                 std::input_iterator_tag = typename std::iterator_traits<It>::iterator_category{}) {
    std::size_t seed = 0;
    for (; first != last; ++first) {
        std::hash<T> hasher;
        _hash_combine_impl(seed, hasher(*first));
    }
    return seed;
}

template <class It, class T = typename std::iterator_traits<It>::value_type>
constexpr void hash_range(std::size_t &seed, It first, It last,
                          std::input_iterator_tag = typename std::iterator_traits<It>::iterator_category{}) {
    for (; first != last; ++first) {
        std::hash<T> hasher;
        _hash_combine_impl(seed, hasher(*first));
    }
}

template <class T>
constexpr std::size_t hash_value(T const &v) {
    std::hash<T> hasher;
    return hasher(v);
}

template <class T, std::size_t N>
constexpr std::size_t hash_value(const T (&x)[N]) {
    return hash_range(x, x + N);
}

template <class T, std::size_t N>
constexpr std::size_t hash_value(T (&x)[N]) {
    return hash_range(x, x + N);
}

template <class T, std::size_t N>
constexpr std::size_t hash_value(std::array<T, N> const &v) {
    return hash_range(v.begin(), v.end());
}

template <class Tup, std::size_t... Is>
constexpr std::size_t _hash_tuple_impl(Tup const &v, std::index_sequence<Is...>) {
    std::size_t seed = 0;
    ((seed ^= hash_value(std::get<Is>(v)) + 0x9e3779b9 + (seed << 6) + (seed >> 2)), ...);
    return seed;
}

template <class T1, class T2>
constexpr std::size_t hash_value(std::pair<T1, T2> const &v) {
    return _hash_tuple_impl(v, std::make_index_sequence<2>{});
}

template <class... Ts>
constexpr std::size_t hash_value(std::tuple<Ts...> const &v) {
    return _hash_tuple_impl(v, std::make_index_sequence<sizeof...(Ts)>{});
}

template <class Ch, class A>
constexpr std::size_t hash_value(std::basic_string<Ch, std::char_traits<Ch>, A> const &v) {
    return hash_range(v.begin(), v.end());
}

template <class Ch, class A>
constexpr std::size_t hash_value(std::basic_string_view<Ch, std::char_traits<Ch>> const &v) {
    return hash_range(v.begin(), v.end());
}

template <class T = void>
struct generic_hash {
    constexpr std::size_t operator()(T const &t) const {
        return hash_value(t);
    }
};

template <>
struct generic_hash<void> {
    template <class T>
    constexpr std::size_t operator()(T const &t) const {
        return hash_value(t);
    }

    using is_transparent = std::true_type;
};

} // namespace _hash_details

using _hash_details::generic_hash;
using _hash_details::hash_combine;
using _hash_details::hash_combine_32;
using _hash_details::hash_combine_64;
using _hash_details::hash_range;
using _hash_details::hash_value;
