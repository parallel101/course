#include <vector>
#include <string>
#include <iostream>
using namespace std;

void print(char const *begptr, char const *endptr) {
    for (char const *ptr = begptr; ptr != endptr; ptr++) {
        char value = *ptr;
        cout << value << endl;
    }
}

int main() {
    vector<char> a = {'h', 'j', 'k', 'l'};
    char const *begptr = a.data();
    char const *endptr = a.data() + a.size();
    print(begptr, endptr);
    return 0;
}
