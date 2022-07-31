#include <string>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <locale>

using namespace std;


int main() {
    string s = "你好";
    wstring ws = L"你好";
    cout << s.size() << endl;
    cout << ws.size() << endl;
}
