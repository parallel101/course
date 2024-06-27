#include <iostream>
#include <optional>
#include <vector>

template <class T>
struct Simple {
    void show(T value) {
        std::cout << value << '\n';
    }
};

template <template <class T> class TT>
struct Bar {
    TT<int> c1i;
    TT<double> c1d;
};

template <class T>
using MyVec = std::vector<T>;

int main() {
    Bar<Simple> a;
    a.c1d.show(3.14);
    a.c1i.show(42);

    Bar<std::optional> b;
    b.c1d = 3.14;
    b.c1i = std::nullopt;

    // Bar<std::vector> c; // 错误 vector 模板是 `template <T, Alloc> class` 而不是 `template <T> class`

    Bar<MyVec> d;
    d.c1d = {3.14, 2.718};
    d.c1i = {42, 37};

    return 0;
}
