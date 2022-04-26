#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

vector<int> holder;

int main() {
    int *p;
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
