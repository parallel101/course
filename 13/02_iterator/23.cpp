#include <vector>
#include <list>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    int b[] = {233, 666, 985, 211};
    vector<int> a = {1, 2, 3, 4, 5, 6};

    cout << "a = " << a << endl;
    a.insert(a.end(), std::begin(b), std::end(b));
    cout << "a = " << a << endl;

    return 0;
}
