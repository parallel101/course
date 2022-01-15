#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <execution>
#include "ticktock.h"

constexpr int N = 1<<12;

struct Grid {
    std::vector<int> p;
    size_t nx;

    Grid(size_t nx, size_t ny)
        : p(nx * ny), nx(nx)
    {
    }

    int &operator()(size_t x, size_t y) {
        return p[y * nx + x];
    }
};

int main() {
    TICK(java);
    int **p = new int *[N]{};
    for (int i = 0; i < N; i++) {
        p[i] = new int[N]{};
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            p[std::rand() % N][std::rand() % N] += 1;
        }
    }
    for (int i = 0; i < N; i++) {
        delete[] p[i];
    }
    delete[] p;
    TOCK(java);

    TICK(cpp);
    Grid g(N, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            g(std::rand() % N, std::rand() % N) += 1;
        }
    }
    TOCK(cpp);
    return 0;
}
