#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

template <class ...T>
constexpr auto sum(T ...t) {
    return (sizeof(T) + ...);
}

template <class T0, class ...T>
void print(T0 const &t0, T const &...t) {
    std::cout << t0;
    ((std::cout << ' ' << t), ...);
    std::cout << std::endl;
}

void func2(int const &t) {
    printf("int const &\n");
}

void func2(int &&t) {
    printf("int &&\n");
}

template <class ...T>
void func1(T &&...t) {
    func2(std::forward<decltype(t)>(t)...);
}

int main() {
    float arr[1<<20]; int i = 0;
    std::generate_n(std::begin(arr), std::size(arr), [&] () { return std::sin(i++); });
    float sum = std::accumulate(std::begin(arr), std::end(arr), 0.f, std::minus{});
    print(sum);
    return 0;
}
