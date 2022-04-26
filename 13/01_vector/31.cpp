#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5};
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.resize(2);
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.resize(5);
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.resize(7);
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    return 0;
}
