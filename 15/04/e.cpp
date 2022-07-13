#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "happy 10th birthday"s;
    cout << "字符串: " << s << endl;
    int n = stoi(s);
    cout << "数字是: " << n << endl;
}
