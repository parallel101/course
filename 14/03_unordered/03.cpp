#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    auto r = b.equal_range(2);
    b.erase(r.first, r.second);
    cout << "删除2以后：" << b << endl;

    return 0;
}
