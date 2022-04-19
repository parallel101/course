#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6};
    cout << a << endl;
    a.resize(4, 233);
    cout << a << endl;
    return 0;
}
