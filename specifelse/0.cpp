#include "ticktock.h"
#include "randint.h"
#include <vector>
#include <algorithm>

#if 0
__attribute__((noinline)) void uppercase(char *p, int n) {
    for (int i = 0; i < n; i++) {
        p[i] = ('a' <= p[i] && p[i] <= 'z') ? p[i] + 'A' - 'a' : p[i];
    }
}
#else
__attribute__((noinline)) void uppercase(char *p, int n) {
    for (int i = 0; i < n; i++) {
        if ('a' <= p[i] && p[i] <= 'z')
            p[i] = p[i] + 'A' - 'a';
    }
}
#endif

int main() {
    int n = (int)1e7;
    std::vector<char> a(n);

    for (int i = 0; i < n; i++) {
        a[i] = randint<char>(0, 127);
    }
    std::sort(a.begin(), a.end());

    TICK(test);
    uppercase(a.data(), n);
    TOCK(test);

    return 0;
}
