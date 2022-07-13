#include <string>
#include <iostream>

using namespace std;

int main() {
    auto s = "42 yuan"s;
    int n = stoi(s);
    cout << n << endl;
}
