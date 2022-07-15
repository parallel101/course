#include <string>
#include <iostream>

using namespace std;

size_t count(string const &str, string const &sub) {
    size_t n = 0, pos = 0;
    while (true) {
        pos = str.find(sub, pos);
        if (pos == str.npos)
            break;
        ++n;
        pos += sub.size();
    }
    return n;
}

int main() {
    cout << count("helloworld,bellreally"s, "ll"s) << endl;
}
