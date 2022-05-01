#include <vector>
#include <list>
#include <iostream>
#include "printer.h"
using namespace std;

int main() {
    int b[] = {233, 666, 985, 211};
    vector<int> a(std::begin(b), std::end(b));

    cout << "a = " << a << endl;

    return 0;
}
