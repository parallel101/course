#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    cout << "从第2个字符开始长为5的子字符串:    " << s.substr(2, 4) << endl;
    cout << "从第2个字符开始长为99的子字符串:   " << s.substr(2, 99) << endl;
    cout << "从第2个字符开始直到末尾的子字符串: " << s.substr(2) << endl;
    cout << "从头开始长为5的子字符串:           " << s.substr(0, 4) << endl;
    cout << "从第100个字符开始长为5的子字符串:  " << s.substr(100, 5) << endl;
}
