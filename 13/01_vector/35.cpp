#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
#include "mallochook.h"
using namespace std;

int main() {
    vector<int> a;
    a.reserve(100);
    for (int i = 0; i < 100; i++)
        a.push_back(i);
    cout << a << endl;
    return 0;
}
