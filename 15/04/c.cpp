#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "42yuan"s;
    size_t pos;
    int n = stoi(s, &pos);
    cout << "原始字符串: " << s << endl;
    cout << "数字部分从第" << pos << "个字符结束" << endl;
    cout << "数字是" << n << endl;
    cout << "剩余的部分是" << s.substr(pos) << endl;
}
