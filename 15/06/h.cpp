#include <string>
#include <iostream>

using namespace std;

void replace_all(string &s, string const &from, string const &to) {
    size_t pos = 0;
    while (true) {
        pos = s.find(from, pos);
        if (pos == s.npos)
            break;
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

int main() {
    string s = "make pyb happy, as pyb said, cihou pyb"s;
    replace_all(s, "pyb", "zhxx");
    cout << s << endl;
}
