#include <vector>
#include <string>
#include <iostream>
using namespace std;

void print(vector<char> const &a) {
    for (int i = 0; i < a.size(); i++) {
        cout << a[i] << endl;
    }
}

void print(string const &a) {
    for (int i = 0; i < a.size(); i++) {
        cout << a[i] << endl;
    }
}

int main() {
    vector<char> a = {'h', 'j', 'k', 'l'};
    print(a);
    string b = {'h', 'j', 'k', 'l'};
    print(b);
    return 0;
}
