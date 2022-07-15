#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    cout << "第0个字符:  " << s[0] << endl;
    cout << "第3个字符:  " << s[3] << endl;
    cout << "第16个字符: " << s[16] << endl;
    cout << "第0个字符:  " << s.at(0) << endl;
    cout << "第3个字符:  " << s.at(3) << endl;
    cout << "第16个字符: " << s.at(16) << endl;
}
