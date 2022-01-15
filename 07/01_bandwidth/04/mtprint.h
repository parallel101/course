#pragma once

#include <iostream>
#include <sstream>

template <class T, class ...Ts>
static void mtprint(T &&t, Ts &&...ts) {
    std::stringstream ss;
    ss << std::forward<T>(t);
    ((ss << ' ' << std::forward<Ts>(ts)), ...);
    ss << std::endl;
    std::cout << ss.str();
}
