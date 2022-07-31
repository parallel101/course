#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

int main() {
    //string s = (stringstream() << setprecision(100) << 3.1415f).str();
    //stringstream("3.1415926535") >> f;
    int n = 10011;
    cout << setprecision(4) << (n * 0.01) << endl;
}
