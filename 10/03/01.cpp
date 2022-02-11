#include "bate.h"

#define N (512*512)

template <int Bshift, class Node>
struct DenseBlock {
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = Bshift;

    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    Node m_data[B][B];

    Node *fetch(int x, int y) const {
        return &m_data[x & Bmask][y & Bmask];
    }

    Node *touch(int x, int y) {
        return &m_data[x & Bmask][y & Bmask];
    }

    template <class Func>
    void foreach(Func const &func) {
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < B; y++) {
                func(x, y, *m_data[x][y]);
            }
        }
    }
};

template <int Bshift, class Node>
struct PointerBlock {
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = Bshift;

    static constexpr int B = 1 << Bshift;
    static constexpr int Bmask = B - 1;

    std::unique_ptr<Node> m_data[B][B];

    Node *fetch(int x, int y) const {
        return m_data[x & Bmask][y & Bmask].get();
    }

    Node *touch(int x, int y) {
        auto &block = m_data[x & Bmask][y & Bmask];
        if (!block)
            block = std::make_unique<Node>();
        return block.get();
    }

    template <class Func>
    void foreach(Func const &func) {
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < B; y++) {
                auto ptr = m_data[x][y].get();
                if (ptr)
                    func(x, y, ptr);
            }
        }
    }
};

template <class Node>
struct HashBlock {
    static constexpr bool isPlace = false;
    static constexpr bool bitShift = 0;

    struct MyHash {
        std::size_t operator()(std::tuple<int, int> const &key) const {
            auto const &[x, y] = key;
            return (x * 2718281828) ^ (y * 3141592653);
        }
    };

    std::unordered_map<std::tuple<int, int>, Node, MyHash> m_data;

    Node *fetch(int x, int y) const {
        auto it = m_data.find(std::make_tuple(x, y));
        if (it == m_data.end())
            return nullptr;
        return &it->second;
    }

    Node *touch(int x, int y) {
        auto it = m_data.find(std::make_tuple(x, y));
        if (it == m_data.end()) {
            auto ptr = std::make_unique<Node>();
            auto rawptr = ptr.get();
            m_data.emplace(std::make_tuple(x, y), std::move(ptr));
            return rawptr;
        }
        return it->second.get();
    }

    template <class Func>
    void foreach(Func const &func) {
        for (auto &[key, block]: m_data) {
            auto &[x, y] = key;
            func(x, y, &block);
        }
    }
};

template <class T>
struct PlaceData {
    static constexpr bool isPlace = true;

    T m_value;

    T read() const {
        return m_value;
    }

    void write(T value) {
        m_value = value;
    }

    template <class Func>
    void visit(Func const &func) {
        func(m_value);
    }
};

template <class T, class Layout>
struct RootGrid {
    Layout m_root;

    template <class Node>
    static T _read(Node const &node, int x, int y) {
        if constexpr (node.isPlace) {
            return node.read();
        } else {
            auto const *child = node.fetch(x >> node.bitShift, y >> node.bitShift);
            if (!child)
                return T{};
            return _read(*child, x, y);
        }
    }

    T read(int x, int y) const {
        return _read(m_root, x, y);
    }

    template <class Node>
    static void _write(Node &node, int x, int y, T value) {
        if constexpr (node.isPlace) {
            return node.write(value);
        } else {
            auto *child = node.touch(x >> node.bitShift, y >> node.bitShift);
            return _write(*child, x, y, value);
        }
    }

    void write(int x, int y, T value) const {
        return _write(m_root, x, y, value);
    }

    template <class Node, class Func>
    static void _foreach(Node &node, int x, int y, Func const &func) {
        if constexpr (node.isPlace) {
            return node.visit([&] (T &val) {
                func(x, y, val);
            });
        } else {
            int xb = x << node.bitShift;
            int yb = y << node.bitShift;
            return node.foreach([&] (int x, int y, auto *child) {
                _foreach(*child, x, y, func);
            });
        }
    }

    template <class Func>
    void foreach(Func const &func) {
        _foreach(m_root, 0, 0, func);
    }
};

using Grid = RootGrid<char, HashBlock<PointerBlock<11, DenseBlock<8, PlaceData<char>>>>>;

int main() {
    bate::timing("main");

    auto *a = new Grid{};

    float px = -100.f, py = 100.f;
    float vx = 0.2f, vy = -0.6f;

    for (int step = 0; step < N; step++) {
        px += vx;
        py += vy;
        int x = (int)std::floor(px);
        int y = (int)std::floor(py);
        a->write(x, y, 1);
    }

    int count = 0;
    a->foreach([&] (int x, int y, char &value) {
        if (value != 0) {
            count++;
        }
    });
    printf("count: %d\n", count);

    bate::timing("main");
    return 0;
}
