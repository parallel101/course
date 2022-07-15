#include <string>
#include <string_view>
#include <iostream>

using namespace std;

int main() {
    string s = "hello";
    string_view sv = s;
    sv = sv.substr(1, 3);
    cout << sv << endl;
}
