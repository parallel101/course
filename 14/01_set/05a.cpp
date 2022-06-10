#include <vector>
#include <set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6, 7};
    vector<int>::iterator a_it1 = a.begin();
    vector<int>::iterator a_it2 = a.end();
    // 会调用 a_it2 - a_it1
    int a_size = std::distance(a_it1, a_it2);
    cout << "vector size = " << a_size << endl;

    set<int> b = {1, 2, 3, 4, 5, 6, 7};
    set<int>::iterator b_it1 = b.begin();
    set<int>::iterator b_it2 = b.end();
    // 会调用 ++b_it1 直到等于 b_it2
    int b_size = std::distance(b_it1, b_it2);
    cout << "set size = " << b_size << endl;

    return 0;
}
