#pragma once

#include <cstdint>
#include <tuple>

namespace morton2d {

constexpr static uint64_t encode1(uint64_t x)
{
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}

constexpr static uint64_t encode(uint64_t x, uint64_t y)
{
    return encode1(x) | encode1(y << 1);
}

constexpr static uint64_t decode1(uint64_t x)
{
    x = x & 0x5555555555555555;
    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0xFFFFFFFFFFFFFFFF;
    return x;
}

constexpr static std::tuple<uint64_t, uint64_t> decode(uint64_t d)
{
    return {decode1(d), decode1(d >> 1)};
}

}

namespace morton3d {

constexpr static uint64_t encode1(uint64_t x)
{
    x = (x | (x << 32)) & 0x7fff00000000ffff; // 0b0111111111111111000000000000000000000000000000001111111111111111
    x = (x | (x << 16)) & 0x00ff0000ff0000ff; // 0b0000000011111111000000000000000011111111000000000000000011111111
    x = (x | (x <<  8)) & 0x700f00f00f00f00f; // 0b0111000000001111000000001111000000001111000000001111000000001111
    x = (x | (x <<  4)) & 0x30c30c30c30c30c3; // 0b0011000011000011000011000011000011000011000011000011000011000011
    x = (x | (x <<  2)) & 0x1249249249249249; // 0b0001001001001001001001001001001001001001001001001001001001001001
    return x;
}

constexpr static uint64_t encode(uint64_t x, uint64_t y, uint64_t z)
{
    return encode1(x) | encode1(y << 1) | encode1(z << 2);
}

constexpr static uint64_t decode1(uint64_t x)
{
    x = x & 0x1249249249249249;
    x = (x | (x >>  2)) & 0x30c30c30c30c30c3;
    x = (x | (x >>  4)) & 0x700f00f00f00f00f;
    x = (x | (x >>  8)) & 0x00ff0000ff0000ff;
    x = (x | (x >> 16)) & 0x7fff00000000ffff;
    x = (x | (x >> 32)) & 0xffffffffffffffff;
    return x;
}

constexpr static std::tuple<uint64_t, uint64_t, uint64_t> decode(uint64_t d)
{
    return {decode1(d), decode1(d >> 1), decode1(d >> 2)};
}

}
