#include <cmath>
#include <type_traits>

void func(float *a) {
    for (int i = 0; i < 1024; i++) {
        auto x = sqrt(a[i]);      // wrong and bad
        static_assert(std::is_same_v<decltype(x), double>);
        auto y = sqrtf(a[i]);     // correct but bad
        static_assert(std::is_same_v<decltype(y), float>);
        auto z = std::sqrt(a[i]); // correct and good
        static_assert(std::is_same_v<decltype(z), float>);
    }
}
