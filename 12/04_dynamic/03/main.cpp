#include <cstdio>
#include <cstdint>
#include <cstdlib>

void fillarr(char* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 42;
    }
}

void printarr(const char* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int main() {
    int n = rand() % 1024;
    char* a = (char*)malloc(n);
    fillarr(a, n);
    printarr(a, n);
    free(a);
    return 0;
}
