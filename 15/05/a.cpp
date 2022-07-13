#include <iostream>
#include <string>

using namespace std;

int main() {
    cout << string("hello\0cpp") << endl;
    cout << string("hello\0cpp", 9) << endl;
}
