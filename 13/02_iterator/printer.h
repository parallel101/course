#pragma once

#include <iostream>
#include <vector>

namespace std {

template <class T>
ostream &operator<<(ostream &os, vector<T> const &v) {
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
