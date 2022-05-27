#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    set<int> b = {1, 4, 2, 1};
    cout << "插入之前: " << b << endl;
    b.insert(3);
    cout << "插入之后: " << b << endl;

    return 0;
}
