#include <iostream>
#include <map>
#include "printer.h"

using namespace std;

int main() {
    map<string, int> m;
    m.at("hello") = 1;
    m.at("world") = 2;
    cout << m << endl;
    return 0;
}
