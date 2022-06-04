#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {0, 1, 3, 4, 5};

    cout << "原始集合：" << b << endl;

    for (auto it = b.begin(); it != b.end(); ++it) {
        int value = *it;
        cout << value << endl;
    }

    return 0;
}
