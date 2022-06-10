#include <vector>
#include <set>
#include <string>
#include "printer.h"

using namespace std;

int main() {
    vector<string> a = {"arch", "any", "zero", "Linux"};
    cout << "vector=" << a << endl;
    set<string> b = {"arch", "any", "zero", "Linux"};
    cout << "set=" << b << endl;
    return 0;
}
