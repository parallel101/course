#include <unordered_set>
#include "printer.h"

using namespace std;

int main() {
    unordered_set<int> a;
    a.reserve(12);
    a.insert(3);
    a.insert(4);
    cout << a.load_factor() << endl;
    cout << a.max_load_factor() << endl;
    return 0;
}
