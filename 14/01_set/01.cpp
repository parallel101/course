#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {9, 8, 5, 2, 1, 1};
    cout << "vector=" << a << endl;
    set<int> b = {9, 8, 5, 2, 1, 1};
    cout << "set=" << b << endl;
    return 0;
}
