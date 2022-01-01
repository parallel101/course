#pragma once

#include <iostream>
#include <chrono>

template <class Name, class Func>
static inline void profile(int times, Name const &name, Func const &func) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < times; i++) {
        func();
    }
    auto t1 = std::chrono::steady_clock::now();
    long dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / times;
    std::cout << name << ": " << dt << std::endl;
}
