#pragma once

#include <iostream>
#include <utility>
#include <type_traits>

namespace std {

template <class T, class = const char *>
struct __printer_test_c_str {
    using type = void;
};

template <class T>
struct __printer_test_c_str<T, decltype(std::declval<T>().c_str())> {};

template <class T, int = 0, int = 0, int = 0,
         class = decltype(std::declval<std::ostream &>() << *++std::declval<T>().begin()),
         class = decltype(std::declval<T>().begin() != std::declval<T>().end()),
         class = typename __printer_test_c_str<T>::type>
std::ostream &operator<<(std::ostream &os, T const &v) {
    os << '{';
    auto it = v.begin();
    if (it != v.end()) {
        os << *it;
        for (++it; it != v.end(); ++it)
            os << ',' << *it;
    }
    os << '}';
    return os;
}

}
