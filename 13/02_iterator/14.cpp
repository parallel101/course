#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6};

    vector<int>::iterator b = a.begin();
    vector<int>::iterator e = a.end();

    cout << "a = " << a << endl;
    cout << "b[0] = " << b[0] << endl;
    cout << "b[1] = " << b[1] << endl;
    cout << "b[2] = " << b[2] << endl;
    cout << "e[-2] = " << e[-2] << endl;
    cout << "e[-1] = " << e[-1] << endl;
    cout << "e[0] = " << e[0] << endl;

    return 0;
}
