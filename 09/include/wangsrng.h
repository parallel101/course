#pragma once

#include <cstdint>
#include <climits>

// Wang's Hash
// https://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// generates determinstic random numbers depending on argument
// unlike the single-threaded std::rand, wangshash can be trivially parallelized
// very useful in Path Tracing etc., where seed can be the pixel coordinates
struct wangsrng {
    uint32_t seed;

    constexpr wangsrng(uint32_t seed = 0) : seed(seed) {}

    constexpr wangsrng(uint32_t seedx, uint32_t seedy)
        : wangsrng(seedx ^ randomize(seedy)) {}

    constexpr wangsrng(uint32_t seedx, uint32_t seedy, uint32_t seedz)
        : wangsrng(seedx ^ randomize(seedy ^ randomize(seedz))) {}

    constexpr static uint32_t randomize(uint32_t i) {
        i = (i ^ 61) ^ (i >> 16);
        i *= 9;
        i ^= i << 4;
        i *= 0x27d4eb2d;
        i ^= i >> 15;
        return i;
    }

    constexpr uint32_t operator()() {
        seed = randomize(seed);
        return seed;
    }

    constexpr uint32_t next_uint32() {
        return operator()();
    }

    constexpr int32_t next_int32() {
        return (int32_t)next_uint32();
    }

    constexpr uint16_t next_uint16() {
        return uint16_t(next_uint32() & 0xffff);
    }

    constexpr int16_t next_int16() {
        return (int16_t)next_uint16();
    }

    constexpr uint8_t next_uint8() {
        return uint8_t(next_uint32() & 0xff);
    }

    constexpr int8_t next_int8() {
        return (int8_t)next_uint8();
    }

    constexpr bool next_bool() {
        return next_uint32() & 1;
    }

    constexpr uint64_t next_uint64() {
        return (uint64_t)next_uint32() | ((uint64_t)next_uint32() << 32);
    }

    constexpr int64_t next_int64() {
        return (int64_t)next_uint64();
    }

    constexpr uintptr_t next_uintptr() {
        if constexpr (sizeof(uintptr_t) == sizeof(uint32_t))
            return (uintptr_t)next_uint32();
        else
            return (uintptr_t)next_uint64();
    }

    constexpr intptr_t next_intptr() {
        if constexpr (sizeof(intptr_t) == sizeof(int32_t))
            return (intptr_t)next_int32();
        else
            return (intptr_t)next_int64();
    }

    constexpr float next_float() {
        return next_uint32() * (1.0f / UINT32_MAX);
    }

    constexpr double next_double() {
        return next_uint64() * (1.0 / UINT64_MAX);
    }
};
