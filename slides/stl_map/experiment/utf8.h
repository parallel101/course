#pragma once

#include <string>
#include <string_view>
#include <numeric>
#include <iostream>
#include <tuple>
#include <algorithm>

inline size_t decode_utf8_len(char8_t cp) {
    if ((cp & 0b1111'1000) == 0b1111'0000) {
        return 4;
    } else if ((cp & 0b1111'0000) == 0b1110'0000) {
        return 3;
    } else if ((cp & 0b1110'0000) == 0b1100'0000) {
        return 2;
    } else {
        return 1;
    }
}

inline std::pair<char8_t const *, char32_t> decode_utf8(char8_t const *in) {
    char32_t cp = static_cast<char32_t>(*in++);
    if ((cp & 0b1000'0000) == 0b0000'0000) {
        /* nothing */
    } else if ((cp & 0b1111'1000) == 0b1111'0000) {
        char32_t cp1 = *in++;
        char32_t cp2 = *in++;
        char32_t cp3 = *in++;
        cp = (cp & 0b0000'0111) << 18
            | (cp1 & 0b0011'1111) << 12
            | (cp2 & 0b0011'1111) << 6
            | (cp3 & 0b0011'1111);
    } else if ((cp & 0b1111'0000) == 0b1110'0000) {
        char32_t cp1 = *in++;
        char32_t cp2 = *in++;
        cp = (cp & 0b0000'1111) << 12
            | (cp1 & 0b0011'1111) << 6
            | (cp2 & 0b0011'1111);
    } else if ((cp & 0b1110'0000) == 0b1100'0000) {
        char32_t cp1 = *in++;
        cp = (cp & 0b0001'1111) << 6
            | (cp1 & 0b0011'1111);
    } else {
        cp = 0xfffd;
    }
    return {in, cp};
}

inline size_t encode_utf8_len(char32_t cp) {
    if (cp >= 0x10000) {
        return 4;
    } else if (cp >= 0x800) {
        return 3;
    } else if (cp >= 0x80) {
        return 2;
    } else {
        return 1;
    }
}

inline char8_t *encode_utf8(char8_t *out, char32_t cp) {
    auto put = [&out] (char32_t x) {
        *out++ = static_cast<char8_t>(x);
    };
    if (cp >= 0x10000) {
        put(0b1111'0000 | cp >> 18);
        put(0b1000'0000 | cp >> 12 & 0b0011'1111);
        put(0b1000'0000 | cp >> 6 & 0b0011'1111);
        put(0b1000'0000 | cp & 0b0011'1111);
    } else if (cp >= 0x800) {
        put(0b1110'0000 | cp >> 12);
        put(0b1000'0000 | cp >> 6 & 0b0011'1111);
        put(0b1000'0000 | cp & 0b0011'1111);
    } else if (cp >= 0x80) {
        put(0b1100'0000 | cp >> 6);
        put(0b1000'0000 | cp & 0b0011'1111);
    } else {
        put(cp & 0b0111'1111);
    }
    return out;
}

inline size_t decode_utf8_string(std::u32string &out, std::u8string_view in) {
    size_t count = 0, rest = 0;
    for (size_t i = 0; i < in.size(); ) {
        size_t n = decode_utf8_len(in[i]);
        if (n <= in.size() - i) {
            i += n;
            ++count;
        } else {
            rest = in.size() - i;
            break;
        }
    }
    out.resize(count);
    char32_t *out_p = out.data();
    char8_t const *in_p = in.data();
    for (size_t i = 0; i < count; i++) {
        char32_t cp;
        std::tie(in_p, cp) = decode_utf8(in_p);
        *out_p++ = cp;
    }
    return rest;
}

inline void encode_utf8_string(std::u8string &out, std::u32string_view in) {
    size_t len = std::transform_reduce(in.begin(), in.end(),
                                       static_cast<size_t>(0), std::plus<size_t>(),
                                       encode_utf8_len);
    out.resize(len);
    std::accumulate(in.begin(), in.end(), out.data(), encode_utf8);
}
