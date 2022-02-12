#include "bate.h"
#include <tbb/spin_mutex.h>
#include <mutex>
#include <atomic>

#define N (512*512)

struct Grid {
    static constexpr int Bshift = 8;
    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    static constexpr int B1shift = 11;
    static constexpr int B1 = 1 << B1shift;
    static constexpr int B1mask = B1 - 1;

    struct Block {
        char m_block[B][B];
    };

    tbb::spin_mutex m_mtx[B1][B1];
    std::unique_ptr<Block> m_data[B1][B1];  // ~1MB

    char read(int x, int y) const {
        auto &block = m_data[(x >> Bshift) & B1mask][(y >> Bshift) & B1mask];
        if (!block)
            return 0;
        return block->m_block[x & Bmask][y & Bmask];
    }

    struct WriteAccessor {
        Grid &that;
        std::map<std::tuple<int, int>, Block *> m_cached;

        void write(int x, int y, char value) {
            auto block = [&] {
                int coorx = (x >> Bshift) & B1mask;
                int coory = (y >> Bshift) & B1mask;
                auto it = m_cached.find({coorx, coory});
                if (it != m_cached.end()) {
                    return it->second;
                }
                auto &block = that.m_data[coorx][coory];
                if (!block) {
                    std::lock_guard _(that.m_mtx[coorx][coory]);
                    if (!block)
                        block = std::make_unique<Block>();
                }
                m_cached.try_emplace({coorx, coory}, block.get());
                return block.get();
            }();
            block->m_block[x & Bmask][y & Bmask] = value;
        }
    };

    WriteAccessor writeAccess() {
        return {*this};
    }

    template <class Func>
    void foreach(Func const &func) {
#pragma omp parallel for collapse(2)
        for (int x1 = 0; x1 < B1; x1++) {
            for (int y1 = 0; y1 < B1; y1++) {
                auto const &block = m_data[x1 & B1mask][y1 & B1mask];
                if (!block)
                    continue;
                int xb = x1 << B1shift;
                int yb = y1 << B1shift;
                for (int dx = 0; dx < B; dx++) {
                    for (int dy = 0; dy < B; dy++) {
                        func(xb | dx, yb | dy, block->m_block[dx][dy]);
                    }
                }
            }
        }
    }
};

int main() {
    bate::timing("main");

    Grid *a = new Grid{};

    float px = -100.f, py = 100.f;
    float vx = 0.2f, vy = -0.6f;

    auto wa_a = a->writeAccess();
#pragma omp parallel for firstprivate(wa_a)
    for (int step = 0; step < N; step++) {
        int x = (int)std::floor(px + vx * step);
        int y = (int)std::floor(py + vy * step);
        wa_a.write(x, y, 1);
    }

    std::atomic<int> count = 0;
    a->foreach([&] (int x, int y, char &value) {
        if (value != 0) {
            count++;
        }
    });
    printf("count: %d\n", count.load());

    bate::timing("main");
    return 0;
}
