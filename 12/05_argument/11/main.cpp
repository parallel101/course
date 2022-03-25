#include <cstdio>
#include <cstdint>
#include <vector>

void func(int arr[6], int size) {
    printf("sizeof(arr): %ld\n", sizeof(arr));
    for (int i = 0; i < 6; i++) {
        printf("%d\n", arr[i]);
    }
}

int main() {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    func(arr, 6);
    return 0;
}
