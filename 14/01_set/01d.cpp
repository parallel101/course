#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include "printer.h"

using namespace std;

struct MyComp {
    bool operator()(string a, string b) const {
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        return a < b;
    }
};

int main() {
    set<string, MyComp> b = {"arch", "any", "zero", "Linux", "linUX"};
    cout << "set=" << b << endl;
    const_cast<string &>(*b.find("any")) = "zebra";
    cout << "set=" << b << endl;
    cout << "found=" << b.count("zebra") << endl;
    return 0;
}
