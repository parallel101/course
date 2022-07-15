#include <string>
#include <string_view>
#include <iostream>

using namespace std;

int main() {
    auto s = "HELLO"s;
    string_view sv = s;
    string news;
    news += sv.substr(0, 2);
    news += "world"sv;
    news += sv.substr(2);
    cout << news << endl;
}
