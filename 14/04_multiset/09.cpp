#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> arr = {9, 8, 5, 2, 1, 1};

    cout << "原始数组：" << arr << endl;

    multiset<int> b(arr.begin(), arr.end());
    arr.assign(b.begin(), b.end());

    cout << "排序后的数组：" << arr << endl;

    return 0;
}
