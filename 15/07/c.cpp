#include <string>
#include <string_view>
#include <iostream>

using namespace std;

int main() {
    string s1 = "hello";
    string_view sv1 = s1;
    s1[0] = 'M';
    cout << sv1 << endl;  // 不会失效
    s1 = "helloworld";
    cout << sv1 << endl;  // 失效!
}
