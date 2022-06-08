#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;

    auto r = b.equal_range(2);
    size_t n = std::distance(r.first, r.second);
    cout << "等于2的元素个数：" << n << endl;

    for (auto it = r.first; it != r.second; ++it) {
        int value = *it;
        cout << value << endl;
    }

    return 0;
}
