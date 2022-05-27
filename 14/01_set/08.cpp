#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    auto res4 = b.insert(4);
    cout << "插入4成功：" << res4.second << endl;
    auto res3 = b.insert(3);
    cout << "插入3成功：" << res3.second << endl;

    return 0;
}
