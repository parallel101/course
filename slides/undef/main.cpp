#define _GLIBCXX_DEBUG
#include <vector>

std::vector<int> a = {1, 2, 3};

void func() {
    a[2];
}

int main() {
    func();
    return 0;
}
