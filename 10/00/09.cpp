#include "bate.h"

#define N (1024*1024)

struct Matrix {
    std::vector<std::tuple<int, float>> m_data;
    int m_row_offset[N]{};

    void createRow(int x, std::vector<std::tuple<int, float>> const &values) {
        m_row_offset[x] = m_data.size();
        for (auto &value: values) {
            m_data.push_back(value);
        }
    }

    template <class Func>
    void foreach(Func &&func) {
        for (int x = 0; x < N; x++) {
            int row_end = x == N - 1 ? m_data.size() : m_row_offset[x + 1];
            for (int offset = m_row_offset[x]; offset < row_end; offset++) {
                auto &[y, value] = m_data[offset];
                func(x, y, value);
            }
        }
    }
};

struct Vector {
    float m_data[N];

    float &at(int x) {
        return m_data[x];
    }
};

int main() {
    bate::timing("main");

    Matrix *a = new Matrix{};
    Vector *v = new Vector{};
    Vector *w = new Vector{};

    for (int i = 0; i < N; i++) {
        v->at(i) = bate::frand();
    }

    a->createRow(0, {{0, 2}, {1, -1}});
    for (int i = 1; i < N - 1; i++) {
        a->createRow(i, {{i - 1, -1}, {i, 2}, {i + 1, -1}});
    }
    a->createRow(N - 1, {{N - 2, -1}, {N - 1, 2}});

    a->foreach([&] (int i, int j, float &value) {
        w->at(i) += value * v->at(j);
    });

    for (int i = 1; i < N - 1; i++) {
        if (std::abs(2 * v->at(i) - v->at(i - 1) - v->at(i + 1) - w->at(i)) > 0.0001f) {
            printf("wrong at %d\n", i);
            return 1;
        }
    }
    printf("all correct\n");

    bate::timing("main");
    return 0;
}
