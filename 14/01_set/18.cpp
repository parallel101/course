#include <vector>
#include <set>
#include "printer.h"

using namespace std;

static void check(bool success) {
    if (!success) throw;
    cout << "通过测试" << endl;
}

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    check(b.find(2) == b.end());
    check(b.lower_bound(2) == b.find(3));
    check(b.upper_bound(2) == b.find(3));

    return 0;
}
