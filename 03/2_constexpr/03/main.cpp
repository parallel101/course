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

constexpr bool isnegative(int n) {
    return n < 0;
}

int main() {
    constexpr bool debug = isnegative(-2014);
    std::cout << sumto<debug>(4) << std::endl;
    return 0;
}
