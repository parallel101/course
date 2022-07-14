#include <string>
#include <iostream>
#include <functional>

using namespace std;

int main() {
    string s = "+03.14e-03"s;
    cout << "字符串: " << s << endl;
    float f = stof(s);
    cout << "浮点数: " << f << endl;
}
