#include <set>
#include <unordered_set>
#include "printer.h"

using namespace std;

int main() {
    set<int> a = {1, 4, 2, 8, 5, 7};
    unordered_set<int> b = {1, 4, 2, 8, 5, 7};

    cout << "set: " << a << endl;
    cout << "unordered_set: " << b << endl;

    return 0;
}
