#include <string>
#include <iostream>

using namespace std;

int main() {
    auto s = "HELLO"s;
    s = s.substr(0, 2) + "world"s + s.substr(2);
    cout << s << endl;
}
