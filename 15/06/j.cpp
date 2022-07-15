#include <string>
#include <iostream>

using namespace std;

int main() {
    auto s = "HELLO"s;
    s.insert(2, "world");
    cout << s << endl;
}
