#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    cout << "元素个数：" << b.size() << endl;

    return 0;
}
