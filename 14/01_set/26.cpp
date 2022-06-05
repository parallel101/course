#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> arr = {9, 8, 5, 2, 1, 1};

    cout << "原始数组：" << arr << endl;

    set<int> b(arr.begin(), arr.end());

    cout << "结果集合：" << b << endl;

    return 0;
}
