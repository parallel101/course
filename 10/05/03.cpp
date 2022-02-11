#include "bate.h"

#define N (1024*1024*256)

int main() {
    bate::timing("main");

    std::vector<int8_t> arr(N / 8);

    for (int ib = 0; ib < N / 8; ib++) {
        int8_t bits = 0;
        for (int it = 0; it < 8; it++) {
            int i = ib * 8 + it;
            int val = i % 2;
            bits |= val << it;
        }
        arr[ib] = bits;
    }

    bate::timing("main");
    return 0;
}
