#include <vector>
#include <list>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6};

    cout << "a = " << a << endl;
    a.assign(4, 233);
    cout << "a = " << a << endl;

    return 0;
}
