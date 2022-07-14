#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

int main() {
    string s = "42yuan"s;
    stringstream ss(s);
    int num;
    ss >> num;
    string unit;
    ss >> unit;
    cout << "数字：" << num << endl;
    cout << "单位：" << unit << endl;
}
