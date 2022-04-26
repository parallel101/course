#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5};
    int *p = a.data();
    cout << p[0] << endl;
    cout << p[0] << endl;
    a.resize(1024);
    cout << p[0] << endl;
    return 0;
}
