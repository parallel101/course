#pragma once

#include <random>

template <class T>
static T randint(T minVal, T maxVal) {
    static std::mt19937 gen(0);
    std::uniform_int_distribution<char> uni(minVal, maxVal);
    return uni(gen);
}

