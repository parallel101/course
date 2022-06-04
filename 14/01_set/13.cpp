#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};

    if (b.count(2)) {
        cout << "集合中存在2" << endl;
    } else {
        cout << "集合中没有2" << endl;
    }

    if (b.count(8)) {
        cout << "集合中存在8" << endl;
    } else {
        cout << "集合中没有8" << endl;
    }

    return 0;
}
