#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 1, 2, 2, 3};

    b.size();
    cout << "原始集合：" << b << endl;
    cout << "等于2的元素个数：" << b.count(2) << endl;
    cout << "等于1的元素个数：" << b.count(1) << endl;

    return 0;
}
