#include <vector>
#include <string>
#include <iostream>
using namespace std;

template <class Ptr>
void print(Ptr begptr, Ptr endptr) {
    for (Ptr ptr = begptr; ptr != endptr; ptr++) {
        auto value = *ptr;
        cout << value << endl;
    }
}

int main() {
    vector<char> a = {'h', 'j', 'k', 'l'};
    char const *abegptr = a.data();
    char const *aendptr = a.data() + a.size();
    print(abegptr, aendptr);
    vector<int> b = {1, 2, 3, 4};
    int const *bbegptr = b.data();
    int const *bendptr = b.data() + b.size();
    print(bbegptr, bendptr);
    return 0;
}
