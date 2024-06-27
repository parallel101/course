// https://en.cppreference.com/w/cpp/numeric/valarray
#include <valarray>

int main() {
    std::valarray<double> a = {0.5, 1.5, 2.5, 3.5};
    std::valarray<double> b = {0.0, 0.1, 0.2, 0.3};

    std::valarray<double> c = (std::sin(a) + 1) * b;
    // 等价于：
    for (size_t i = 0; i < a.size(); ++i) {
        c[i] = (std::sin(a[i]) + 1) * b[i];
    }

    double sum = a.sum();
    // 等价于：
    sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i];
    }

    a = a.shift(1);
    // 等价于：
    std::valarray<double> temp = a;
    for (size_t i = 0; i < a.size() - 1; ++i) {
        a[i] = temp[i + 1];
    }
    a[a.size() - 1] = 0;

    return (int)sum;
}
