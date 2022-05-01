#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6};
    vector<int> b = {233, 666, 985, 211};

    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
    a.insert(a.begin(), b.begin(), b.end());
    cout << "a = " << a << endl;

    return 0;
}
