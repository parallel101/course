#include <vector>
#include <iostream>
using namespace std;

struct C {
    vector<int> a{4};
};

int main() {
    C c;
    cout << "c.a[0] = " << c.a[0] << endl;
    cout << "c.a.size() = " << c.a.size() << endl;
    return 0;
}
