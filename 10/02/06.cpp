#include "bate.h"

#define N (512*512)

struct Grid {
    struct MyHash {
        std::size_t operator()(std::tuple<int, int> const &key) const {
            auto const &[x, y] = key;
            return (x * 2718281828) ^ (y * 3141592653);
        }
    };

    static constexpr int B = 16;

    struct Block {
        char m_block[B][B / 8];
    };

    std::unordered_map<std::tuple<int, int>, Block, MyHash> m_data;  // ~128KB

    char read(int x, int y) const {
        auto it = m_data.find(std::make_tuple(x / B, y / B));
        if (it == m_data.end()) {
            return 0;
        }
        return it->second.m_block[x % B][(y % B) / 8] & (1 << ((y % B) % 8));
    }

    void write(int x, int y, char value) {
        Block &block = m_data[std::make_tuple(x / B, y / B)];
        char &ref = block.m_block[x % B][(y % B) / 8];
        if (value)
            ref |= 1 << ((y % B) % 8);
        else
            ref &= ~(1 << ((y % B) % 8));
    }

    template <class Func, bool modified = false>
    void foreach(Func const &func) {
        for (auto &[key, block]: m_data) {
            auto &[xb, yb] = key;
            for (int dx = 0; dx < B; dx++) {
                for (int dy = 0; dy < B; dy++) {
                    char value = block.m_block[dx][dy / 8] & (1 << (dy % 8));
                    func(xb * B + dx, yb * B + dy, value);
                    if constexpr (modified) {
                        char &ref = block.m_block[dx][dy / 8];
                        if (value)
                            ref |= 1 << (dx % 8);
                        else
                            ref &= ~(1 << (dx % 8));
                    }
                }
            }
        }
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
    a->foreach([&] (int x, int y, char value) {
        if (value != 0) {
            count++;
        }
    });
    printf("count: %d\n", count);

    bate::timing("main");
    return 0;
}
