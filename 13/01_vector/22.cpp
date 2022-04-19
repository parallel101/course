#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3};
    cout << a << endl;
    a.pop_back();
    cout << a << endl;
    return 0;
}
