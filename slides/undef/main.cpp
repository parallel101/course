#define _GLIBCXX_DEBUG
#include <vector>

constexpr auto func() {
    return -2 >> 1;
}

int main() {
    constexpr auto _ = func();
    return 0;
}
