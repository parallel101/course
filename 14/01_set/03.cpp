#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    vector<int>::iterator a_it = a.begin() + 3;
    cout << "vector[3]=" << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    set<int>::iterator b_it = b.begin() + 3;
    cout << "set[3]=" << *b_it << endl;

    return 0;
}
