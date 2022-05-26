#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    vector<int>::iterator a_it = a.begin();
    cout << "vector[0]=" << *a_it << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    set<int>::iterator b_it = b.begin();
    cout << "set[0]=" << *b_it << endl;

    return 0;
}
