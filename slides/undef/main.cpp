#define _GLIBCXX_DEBUG
#include <vector>

std::vector<int> v = {1, 2, 3};

void func() {
    auto it = v.begin();
    v.push_back(4); // push_back 可能导致扩容，使元素全部移动到了新的一段内存，会使之前保存的迭代器失效
    *it = 0;        // 错！
}

int main() {
    func();
    return 0;
}
