#include <cstdio>
#include <cstdint>
#include <cstdlib>

int* makearr() {
    int a[1024];
    for (int i = 0; i < 1024; i++)
        a[i] = i;
    return a;
}

int main() {
    int* a = makearr();
    for (int i = 0; i < 1024; i++)
        a[i] += 1;
    return 0;
}
