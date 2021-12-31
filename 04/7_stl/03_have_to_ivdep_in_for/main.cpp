#include <vector>

void func(std::vector<int> &a,
          std::vector<int> &b) {
#pragma GCC ivdep
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] + 1;
    }
}
