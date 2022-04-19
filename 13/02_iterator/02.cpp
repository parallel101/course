#include <vector>
#include <string>
#include <iostream>
using namespace std;

void print(vector<char> const &a) {
    for (int i = 0; i < a.size(); i++) {
        cout << a[i] << endl;
    }
}

int main() {
    vector<char> a = {'h', 'j', 'k', 'l'};
    print(a);
    return 0;
}
