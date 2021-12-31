#include <vector>

void func(std::vector<int> &a,
          std::vector<int> &b,
          std::vector<int> &c) {
    c[0] = a[0];
    c[0] = b[0];
}
