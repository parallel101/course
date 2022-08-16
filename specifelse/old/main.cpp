#include "ticktock.h"
#include "randint.h"
#include <vector>
#include <algorithm>

__attribute__((noinline)) void uppercase_slow(char *p, int n) {
    for (int i = 0; i < n; i++) {
        if ('a' <= p[i] && p[i] <= 'z')
            p[i] = p[i] + 'A' - 'a';
    }
    //random: 0.020724s
    //sorted: 0.004538s
}

__attribute__((noinline)) void uppercase_fast(char *p, int n) {
    for (int i = 0; i < n; i++) {
        p[i] = ('a' <= p[i] && p[i] <= 'z') ? (p[i] + 'A' - 'a') : p[i];
    }
    //random: 0.000735s (28x faster)
    //sorted: 0.000774s
}

#define uppercase uppercase_fast

int main() {
    int n = (int)1e7;
    std::vector<char> a(n);

    for (int i = 0; i < n; i++) {
        a[i] = randint<char>(0, 127);
    }

    TICK(random);
    uppercase(a.data(), n);
    TOCK(random);

    for (int i = 0; i < n; i++) {
        a[i] = randint<char>(0, 127);
    }
    std::sort(a.begin(), a.end());

    TICK(sorted);
    uppercase(a.data(), n);
    TOCK(sorted);

    return 0;
}
