#include <array>

template <int N>
constexpr int func_impl() {
    std::array<int, N> arr{};
    for (int i = 1; i <= N; i++) {
        arr[i - 1] = i;
    }
    int ret = 0;
    for (int i = 1; i <= N; i++) {
        ret += arr[i - 1];
    }
    return ret;
}

int func() {
    constexpr int ret = func_impl<50000>();
    return ret;
}
