#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    cout << boolalpha;
    cout << "lo: " << s.find("lo") << endl;
    cout << "wo: " << s.find("wo") << endl;
    cout << "ma: " << s.find("ma") << endl;
}
