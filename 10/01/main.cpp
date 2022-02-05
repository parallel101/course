#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdlib>
#include <array>

#define N 1024

struct Matrix {
    float m_data[N][N];

    float &at(int x, int y) {
        return m_data[x][y];
    }
};

struct Vector {
    float m_data[N];

    float &at(int x) {
        return m_data[x];
    }
};

int main() {
    Matrix *a = new Matrix{};
    Vector *v = new Vector{};
    Vector *w = new Vector{};

    std::mt19937 gen{1};
    std::uniform_real_distribution<float> unif;

    for (int i = 0; i < N; i++) {
        v->at(i) = unif(gen);
    }

    for (int i = 0; i < N; i++) {
        a->at(i, i) = 2;
        if (i > 0)
            a->at(i - 1, i) = -1;
        if (i < N - 1)
            a->at(i + 1, i) = -1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            w->at(i) += a->at(i, j) * v->at(j);
        }
    }

    printf("v = [\n");
    for (int i = 0; i < N; i++) {
        printf("%.04f%c", v->at(i), " \n"[i % 32 == 31 && i != N - 1]);
    }
    printf("\n]\n");
}
