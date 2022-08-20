#pragma once

#include <random>

template <class T>
static T randint(T minVal, T maxVal) {
    static std::mt19937 gen(0);
    std::uniform_int_distribution<T> uni(minVal, maxVal);
    return uni(gen);
}

template <class T, T minVal, T maxVal>
static T randint() {
    static std::mt19937 gen(0);
    std::uniform_int_distribution<T> uni(minVal, maxVal);
    return uni(gen);
}
