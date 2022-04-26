#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4, 5};
    int *p = a.data();
    int n = a.size();
    memset(p, -1, sizeof(int) * n);
    cout << a << endl;
    return 0;
}
