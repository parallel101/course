#include <iostream>
#include <map>
#include "printer.h"

using namespace std;

int main() {
    map<string, int> m;
    m["hello"] = 1;
    m["world"] = 2;
    cout << m << endl;
    return 0;
}
