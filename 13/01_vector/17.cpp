#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2};
    cout << a << endl;
    a.resize(4);
    cout << a << endl;
    return 0;
}
