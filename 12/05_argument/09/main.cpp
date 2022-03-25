#include <cstdio>
#include <cstdint>
#include <cstdlib>

int* makearr() {
    int* a = (int*)malloc(1024 * sizeof(int));
    for (int i = 0; i < 1024; i++)
        a[i] = i;
    return a;
}

int main() {
    int* a = makearr();
    for (int i = 0; i < 1024; i++)
        a[i] += 1;
    free(a);
    return 0;
}
