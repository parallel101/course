#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

int main() {
    stringstream ss;
    ss << "十六进制：" << hex << 42;
    string s = ss.str();
    cout << s << endl;
}
