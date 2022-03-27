#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> a(4);
    cout << "a.at(0) = " << a.at(0) << endl;
    cout << "a.at(1) = " << a.at(1) << endl;
    cout << "a.at(2) = " << a.at(2) << endl;
    cout << "a.at(3) = " << a.at(3) << endl;
    cout << "a.at(1000) = " << a.at(1000) << endl;
    return 0;
}
