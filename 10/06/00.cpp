#include "bate.h"

#define N (1024*1024*256)

int main() {
    bate::timing("main");

    std::vector<double> arr(N);

    for (int i = 0; i < N; i++) {
        arr[i] = i * 3.14;
    }

    bate::timing("main");
    return 0;
}
