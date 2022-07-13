#include <string>
#include <string_view>
#include <iostream>

using namespace std;

int main() {
    string s1 = "hello";
    string s2 = s1;         // 深拷贝
    string_view sv1 = s1;   // 弱引用
    string_view sv2 = sv1;  // 浅拷贝
    cout << "s1=" << s1 << endl;
    cout << "s2=" << s2 << endl;
    cout << "sv1=" << sv1 << endl;
    cout << "sv2=" << sv2 << endl;
    s1[0] = 'B';
    cout << "修改s1后" << endl;
    cout << "s1=" << s1 << endl;
    cout << "s2=" << s2 << endl;
    cout << "sv1=" << sv1 << endl;
    cout << "sv2=" << sv2 << endl;
}
