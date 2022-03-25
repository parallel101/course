#include <cstdio>
#include <cstdint>
#include <cstdlib>

void fillarr(int* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = 42;
    }
}

void printarr(const int* a, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int main() {
    size_t n = rand() % 1024;
    int* a = (int*)malloc(n * 4);
    fillarr(a, n);
    printarr(a, n);
    free(a);
    return 0;
}
