#include <vector>
#include <list>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5, 6};

    cout << "a = " << a << endl;
    a.assign({233, 666, 985, 211});
    cout << "a = " << a << endl;
    a = {996, 007};
    cout << "a = " << a << endl;
    cout << "a.capacity() = " << a.capacity() << endl;
    a = vector<int>{996, 007};
    cout << "a.capacity() = " << a.capacity() << endl;

    return 0;
}
