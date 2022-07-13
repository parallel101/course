#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "4399cc"s;
    cout << "原始字符串" << s << endl;
    cout << "作为十六进制" << stoi(s, nullptr, 16) << endl;  // 0x4399cc
    cout << "作为十进制" << stoi(s, nullptr, 10) << endl;    // 4399
    cout << "作为八进制" << stoi(s, nullptr, 8) << endl;     // 043
}
