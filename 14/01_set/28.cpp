#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    b.clear();
    // b = {};
    // b.erase(b.begin(), b.end());

    cout << "清空后集合：" << b << endl;

    return 0;
}
