#include "bate.h"
#include <sys/mman.h>

#define N (intptr_t(1)<<20)

int main() {
    bate::timing("main");

    size_t size = N * N * sizeof(float);
    float *arr = (float *)mmap(0, size,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                               -1, 0);
    if (arr == (void *)-1) {
        perror("mmap");
        return -1;
    }

    for (intptr_t y = 4096; y < 8192; y++) {
        for (intptr_t x = 2048; x < 4096; x++) {
            arr[y * N + x] = ((x + y) % 2) * 3.14f;
        }
    }

    munmap(arr, size);

    bate::timing("main");
    return 0;
}
