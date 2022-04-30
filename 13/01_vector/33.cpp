#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5};
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.resize(12);
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.resize(4);
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    a.shrink_to_fit();
    cout << a.data() << ' ' << a.size() << '/' << a.capacity() << endl;
    return 0;
}
