#include "bate.h"

#define N (1024*1024)

struct Matrix {
    std::map<std::tuple<int, int>, float> m_data;

    float read(int x, int y) const {
        auto it = m_data.find(std::make_tuple(x, y));
        if (it != m_data.end()) {
            return it->second;
        }
        return 0;
    }

    void write(int x, int y, float value) {
        auto it = m_data.find(std::make_tuple(x, y));
        if (it != m_data.end()) {
            it->second = value;
        }
    }

    void create(int x, int y, float value) {
        m_data.emplace(std::make_tuple(x, y), value);
    }

    template <class Func>
    void foreach(Func &&func) {
        for (auto &[key, value]: m_data) {
            auto &[x, y] = key;
            func(x, y, value);
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

    for (int i = 0; i < N; i++) {
        a->create(i, i, 2);
        if (i > 0)
            a->create(i - 1, i, -1);
        if (i < N - 1)
            a->create(i + 1, i, -1);
    }

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
