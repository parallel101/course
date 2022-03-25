#include <cstdio>
#include <cstdint>
#include <array>

void func(std::array<int, 6> arr, int size) {
    for (int i = 0; i < arr.size(); i++) {
        printf("%d\n", arr[i]);
    }
}

int main() {
    std::array<int, 6> arr = {1, 2, 3, 4, 5, 6};
    func(arr, 6);
    return 0;
}
