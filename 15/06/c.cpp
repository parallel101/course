#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    cout << "寻找h的结果：" << s.find('h') << endl;
    cout << "寻找e的结果：" << s.find('e') << endl;
    cout << "寻找l的结果：" << s.find('l') << endl;
    cout << "寻找o的结果：" << s.find('o') << endl;
    cout << "寻找H的结果：" << s.find('H') << endl;
    cout << "(size_t) -1：" << (size_t)-1 << endl;
}
