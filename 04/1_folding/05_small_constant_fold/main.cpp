#include <array>

int func() {
    std::array<int, 10> arr;
    for (int i = 1; i <= 10; i++) {
        arr[i - 1] = i;
    }
    int ret = 0;
    for (int i = 1; i <= 10; i++) {
        ret += arr[i - 1];
    }
    return ret;
}
