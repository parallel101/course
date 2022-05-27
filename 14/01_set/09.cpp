#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    auto res1 = b.insert(3);
    cout << "插入3成功：" << res1.second << endl;
    cout << "3所在的位置：" << *res1.first << endl;
    auto res2 = b.insert(3);
    cout << "再次插入3成功：" << res2.second << endl;
    cout << "3所在的位置：" << *res2.first << endl;

    return 0;
}
