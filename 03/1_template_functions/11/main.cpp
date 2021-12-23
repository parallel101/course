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

int main() {
    constexpr bool debug = true;
    std::cout << sumto<debug>(4) << std::endl;
    return 0;
}
