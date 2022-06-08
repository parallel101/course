#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    b.erase(b.lower_bound(2), b.upper_bound(2));
    cout << "删除2以后：" << b << endl;

    return 0;
}
