#include <string>
#include <vector>
#include <iostream>

using namespace std;

vector<string> split(string s) {
    vector<string> ret;
    size_t pos = 0;
    while (true) {
        size_t newpos = s.find_first_of(" \t\v\f\n\r", pos);
        if (newpos == s.npos) {
            ret.push_back(s.substr(pos, newpos));
            break;
        }
        ret.push_back(s.substr(pos, newpos - pos));
        pos = newpos + 1;
    }
    return ret;
}

int main() {
    string s = "hello world\tpyb teacher\ngood job"s;
    vector<string> v = split(s);
    for (auto const &vi: v) {
        cout << vi << endl;
    }
    cout << sizeof(string) << endl;
}
