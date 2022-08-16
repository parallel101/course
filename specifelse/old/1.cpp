#include "ticktock.h"
#include "randint.h"
#include <vector>
#include <algorithm>

__attribute__((noinline)) void uppercase(char *p, int n) {
    for (int i = 0; i < n; i++) {
        if ('a' <= p[i] && p[i] <= 'z')
            p[i] = p[i] + 'A' - 'a';
    }
}

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
