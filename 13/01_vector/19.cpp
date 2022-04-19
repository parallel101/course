#include <vector>
#include <iostream>
#include "printer.h"
using namespace std;

template <class T>
struct test {
    typename decay<T>::type t;

    int func() {
        t->template cast<int>();
    }
};

int main() {
    vector<int> a = {1, 2};
    cout << a << endl;
    a.resize(4, 233);
    cout << a << endl;
    return 0;
}
