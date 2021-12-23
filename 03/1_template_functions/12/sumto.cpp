#include "sumto.h"
#include <iostream>

template <bool debug>
int sumto(int n) {
    int res = 0;
    for (int i = 1; i <= n; i++) {
        res += i;
        if constexpr (debug)
            std::cout << i << "-th: " << res << std::endl;
    }
    return res;
}

template int sumto<true>(int n);
template int sumto<false>(int n);
