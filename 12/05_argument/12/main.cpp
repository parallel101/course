#include <cstdio>
#include <cstdint>
#include <vector>

void func(int arr[], int size) {
    for (int i = 0; i < sizeof(arr) / sizeof(arr[0]); i++) {
        printf("%d\n", arr[i]);
    }
}

int main() {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    func(arr, 6);
    return 0;
}
