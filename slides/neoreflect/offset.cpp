#include <cstddef>
#include <iostream>

struct Y {
    double a, b, c;
};

struct Z : Y {
    double d, e, f;
};

size_t offset() {
    static_assert(std::is_aggregate_v<Y>);
    Y &o = *(Y *)1;
    auto &[x, y, z] = o;
    return (char *)&y - (char *)&o;
}

int main() {
    auto i = offset();
    std::cout << i << '\n';
    auto p = &Y::b;
    int *q = (int *)&p;
    static_assert(sizeof(p) == sizeof(void *));
    std::cout << 'q' << *q << '\n';
    reinterpret_cast<double Y::*>(reinterpret_cast<void *>(0));
    return 0;
}
