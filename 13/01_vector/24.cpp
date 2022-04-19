#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3};
    cout << "a[0] = " << a[0] << endl;
    cout << "a[a.size() - 1] = " << a[a.size() - 1] << endl;
    cout << "a.front() = " << a.front() << endl;
    cout << "a.back() = " << a.back() << endl;
    return 0;
}
