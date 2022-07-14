#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    cout << boolalpha;
    cout << "h: " << (s.find('h') != string::npos) << endl;
    cout << "z: " << (s.find('z') != string::npos) << endl;
}
