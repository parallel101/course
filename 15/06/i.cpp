#include <string>
#include <iostream>

using namespace std;

int main() {
    string s;

    s = "hello, "s;
    s.append("world"s);    // "hello, world"
    cout << s << endl;

    s = "hello, "s;
    s.append("world");     // "hello, world"
    cout << s << endl;

    s = "hello, "s;
    s.append("world"s, 3); // "hello, ld"
    cout << s << endl;

    s = "hello, "s;
    s.append("world", 3);  // "hello, wor"
    cout << s << endl;
}
