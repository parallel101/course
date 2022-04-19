#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3};
    cout << a << endl;
    int val = a.back();
    a.pop_back();
    cout << "back = " << val << endl;
    cout << a << endl;
    return 0;
}
