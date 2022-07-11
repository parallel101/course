#include <cstdio>
#include <string>
#include <iostream>

using namespace std;

string operator""_s(const char *s, size_t len) {
    return string(s, len);
}

int main() {
    string s3 = "hello"_s + "world"_s;
    cout << s3 << endl;
}
