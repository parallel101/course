#include "bate.h"
#include <stdlib.h>

#define N (intptr_t(1)<<20)

int main() {
    bate::timing("main");

    float *arr = (float *)malloc(N * N * sizeof(float));
    if (!arr) {
        perror("malloc");
        return -1;
    }

    for (intptr_t y = 4096; y < 8192; y++) {
        for (intptr_t x = 2048; x < 4096; x++) {
            arr[y * N + x] = ((x + y) % 2) * 3.14f;
        }
    }

    free(arr);

    bate::timing("main");
    return 0;
}
