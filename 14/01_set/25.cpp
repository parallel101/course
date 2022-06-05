#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    vector<int> arr(b.begin(), b.end());

    cout << "结果数组：" << arr << endl;

    return 0;
}
