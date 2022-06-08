#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> b = {0, 1, 3, 4, 5};

    cout << "原始数组：" << b << endl;

    set<int> arr(b.begin(), b.end());

    cout << "结果集合：" << arr << endl;

    return 0;
}
