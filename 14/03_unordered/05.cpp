#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    auto r = b.equal_range(6);

    cout << boolalpha;
    cout << (r.first == b.end()) << endl;
    cout << (r.second == b.end()) << endl;

    return 0;
}
