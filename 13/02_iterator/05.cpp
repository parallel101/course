#include <vector>
#include <string>
#include <iostream>
using namespace std;

void print(char const *a, size_t n) {
    for (int i = 0; i < n; i++) {
        cout << a[i] << endl;
    }
}

int main() {
    vector<char> a = {'h', 'j', 'k', 'l'};
    print(a.data(), a.size() - 1);
    return 0;
}
