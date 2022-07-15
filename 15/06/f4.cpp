#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main() {
    string s = "hello world, pyb teacher? good job!"s;
    int n = s.find_first_not_of(" abcdefghijklmnopqrstuvwxyz");
    cout << n << endl;
}
