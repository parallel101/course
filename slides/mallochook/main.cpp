#include <cstdio>

struct alignas(64) CachelineAligned {
    int x;
};

int main() {
    auto *p1 = new CachelineAligned[1];
    p1->x = 2;
    delete[] p1;
}
