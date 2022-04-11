#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a;
    cout << a << endl;
    a.resize(4, 233);
    cout << a << endl;
    return 0;
}
