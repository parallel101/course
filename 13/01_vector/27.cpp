#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

int main() {
    int *p;
    vector<int> holder;
    {
        vector<int> a = {1, 2, 3, 4, 5};
        p = a.data();
        cout << p[0] << endl;
        cout << p[0] << endl;
        holder = std::move(a);
    }
    cout << p[0] << endl;
    return 0;
}
