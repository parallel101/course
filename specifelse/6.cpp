#include <vector>
#include <algorithm>
#include <map>
#include "ticktock.h"
#include "randint.h"

static int ifelse_magic(int x) {
    if (x == 0) {
        return 233;
    } else if (x == 1) {
        return 42;
    } else if (x == 2) {
        return 666;
    } else if (x == 3) {
        return 985;
    } else {
        return 211;
    }
}

static int switch_magic(int x) {
    switch (x) {
    case 0:
        return 233;
    case 1:
        return 42;
    case 2:
        return 666;
    case 3:
        return 985;
    default:
        return 211;
    }
}

static int map_magic(int x) {
    static const std::map<int, int> lut = {
        {0, 233},
        {1, 42},
        {2, 666},
        {3, 985},
        {4, 211},
    };
    return lut.at(x);
}

static int array_magic(int x) {
    static const int lut[] = {
        233,
        42,
        666,
        985,
        211,
    };
    return lut[x];
}

__attribute__((noinline)) void test(int *a, int n, int (*magic)(int)) {
    for (int i = 0; i < n; i++) {
        a[i] = magic(a[i]);
    }
}

int main() {
    std::vector<int> a((int)1e7);

    std::generate(a.begin(), a.end(), randint<int, 0, 4>);
    TICK(ifelse);
    test(a.data(), a.size(), ifelse_magic);
    TOCK(ifelse);

    std::generate(a.begin(), a.end(), randint<int, 0, 4>);
    TICK(switch);
    test(a.data(), a.size(), switch_magic);
    TOCK(switch);

    std::generate(a.begin(), a.end(), randint<int, 0, 4>);
    TICK(map);
    test(a.data(), a.size(), map_magic);
    TOCK(map);

    std::generate(a.begin(), a.end(), randint<int, 0, 4>);
    TICK(array);
    test(a.data(), a.size(), array_magic);
    TOCK(array);
}
