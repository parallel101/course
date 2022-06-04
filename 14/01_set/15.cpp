#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "原始集合：" << b << endl;
    b.erase(b.find(4));
    cout << "删除元素4：" << b << endl;
    b.erase(b.begin());
    cout << "删最小元素：" << b << endl;
    b.erase(std::prev(b.end()));
    cout << "删最大元素：" << b << endl;

    return 0;
}
