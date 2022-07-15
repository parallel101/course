#include <string>
#include <iostream>

using namespace std;

int main() {
    string s = "helloworld"s;
    int n = s.find_first_of("onl");
    cout << n << endl;
}
