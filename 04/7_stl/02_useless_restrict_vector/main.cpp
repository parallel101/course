#include <vector>

void func(std::vector<int> &__restrict a,
          std::vector<int> &__restrict b,
          std::vector<int> &__restrict c) {
    c[0] = a[0];
    c[0] = b[0];
}
