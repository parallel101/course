#include <iostream>
#include <vector>
#include <list>
using namespace std;

template <class Ptr>
void print(Ptr begptr, Ptr endptr) {
    for (Ptr ptr = begptr; ptr != endptr; ptr++) {
        auto value = *ptr;
        cout << value << endl;
    }
}

int main() {
    list<char> a = {'h', 'j', 'k', 'l'};
    list<char>::iterator begptr = a.begin();
    list<char>::iterator endptr = a.end();
    print(begptr, endptr);
    return 0;
}
