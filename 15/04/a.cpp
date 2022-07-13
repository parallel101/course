#include <string>
#include <iostream>

using namespace std;

int main() {
    int n = 42;
    auto s = to_string(n) + " yuan"s;
    cout << s << endl;
}
