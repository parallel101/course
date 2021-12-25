#include <iostream>
#include <vector>

template <class T>
void print(std::vector<T> const &a) {
    std::cout << "{";
    for (size_t i = 0; i < a.size(); i++) {
        std::cout << a[i];
        if (i != a.size() - 1)
            std::cout << ", ";
    }
    std::cout << "}" << std::endl;
}

int main() {
    std::vector<int> a = {1, 4, 2, 8, 5, 7};
    print(a);
    std::vector<double> b = {3.14, 2.718, 0.618};
    print(b);
    return 0;
}
