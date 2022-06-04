#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 2, 3, 4, 5};

    cout << "删之前：" << b << endl;
    int num = b.erase(4);
    cout << "删之后：" << b << endl;
    cout << "删了 " << num << " 个元素" << endl;

    return 0;
}
