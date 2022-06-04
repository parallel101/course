#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    b.erase(b.lower_bound(2), b.upper_bound(4));
    cout << "删除[2,4]之间的元素：" << b << endl;

    return 0;
}
