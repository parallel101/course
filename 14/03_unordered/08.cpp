#include <set>
#include "printer.h"

using namespace std;

int main() {
    multiset<int> b = {1, 1, 1, 2, 2, 3};

    cout << "原始集合：" << b << endl;
    cout << boolalpha;
    cout << "集合中存在2：" << (b.find(2) != b.end()) << endl;
    cout << "集合中存在1：" << (b.find(1) != b.end()) << endl;
    cout << "第一个1在头部：" << (b.find(1) == b.begin()) << endl;

    return 0;
}
