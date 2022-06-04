#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    auto it = b.find(2);
    cout << "2所在位置：" << *it << endl;
    cout << "比2小的数：" << *prev(it) << endl;
    cout << "比2大的数：" << *next(it) << endl;

    return 0;
}
