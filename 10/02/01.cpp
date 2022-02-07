#include "bate.h"

#define N (512*512)

struct Grid {
    char m_data[N][N];  // 64GB

    char read(int x, int y) const {
        return m_data[x][y];
    }

    void write(int x, int y, char value) {
        m_data[x][y] = value;
    }
};

int main() {
    bate::timing("main");

    Grid *a = new Grid{};

    float px = 0.f, py = 0.f;
    float vx = 0.2f, vy = 0.6f;

    for (int step = 0; step < N; step++) {
        px += vx;
        py += vy;
        int x = (int)std::floor(px);
        int y = (int)std::floor(py);
        a->write(x, y, 1);
    }

    int count = 0;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            if (a->read(x, y) != 0) {
                count++;
            }
        }
    }
    printf("count: %d\n", count);

    bate::timing("main");
    return 0;
}
