#include <vector>
#include <algorithm>
#include "ticktock.h"
#include "randint.h"

static int ifelse_clamp(int x) {
    if (x < 0) {
        return 0;
    } else if (x > 255) {
        return 255;
    } else {
        return x;
    }
}

static int bailan_clamp(int x) {
    return (x < 0) ? 0 : ((x > 255) ? 255 : x);
}

static int addmul_clamp(int x) {
    return (x >= 0) * (x <= 255) * x + (x > 255);
}

__attribute__((noinline)) void test(int *a, int n, int (*clamp)(int)) {
    for (int i = 0; i < n; i++) {
        a[i] = clamp(a[i]);
    }
}

int main() {
    std::vector<int> a((int)1e7);

    std::generate(a.begin(), a.end(), randint<int, -512, 512>);
    TICK(ifelse);
    test(a.data(), a.size(), ifelse_clamp);
    TOCK(ifelse);

    std::generate(a.begin(), a.end(), randint<int, -512, 512>);
    TICK(bailan);
    test(a.data(), a.size(), bailan_clamp);
    TOCK(bailan);

    std::generate(a.begin(), a.end(), randint<int, -512, 512>);
    TICK(addmul);
    test(a.data(), a.size(), addmul_clamp);
    TOCK(addmul);
}
