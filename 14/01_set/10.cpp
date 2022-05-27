#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    auto [ok1, it1] = b.insert(3);
    cout << "插入3成功：" << ok1 << endl;
    cout << "3所在的位置：" << *it1 << endl;
    auto [ok2, it2] = b.insert(3);
    cout << "再次插入3成功：" << ok2 << endl;
    cout << "3所在的位置：" << *it2 << endl;

    return 0;
}
