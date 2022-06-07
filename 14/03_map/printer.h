#pragma once

#include <iostream>
#include <utility>

namespace std {

template <class T,
         class = decltype(std::declval<ostream &>() <<
                          *++std::declval<T>().begin()),
         class = decltype(std::declval<T>().begin() !=
                          std::declval<T>().end())>
ostream &operator<<(ostream &os, T const &v) {
    os << '{';
    auto it = v.begin();
    if (it != v.end()) {
        os << *it;
        for (++it; it != v.end(); ++it) {
            os << ',' << *it;
        }
    }
    os << '}';
    return os;
}

}
