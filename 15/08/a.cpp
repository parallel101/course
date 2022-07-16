#include <string>
#include <vector>
#include <string_view>
#include <iostream>

using namespace std;

struct Empty {
};

struct NonEmpty {
    char c;
};

struct DerivedEmpty : Empty {
    int i;
};

struct DerivedNonEmpty : NonEmpty {
    int i;
};

int main() {
    cout << sizeof(Empty) << endl;            // 1 = 1(empty)
    cout << sizeof(NonEmpty) << endl;         // 1 = 1(char c)
    cout << sizeof(DerivedEmpty) << endl;     // 4 = 0(empty) + 4(int i)
    cout << sizeof(DerivedNonEmpty) << endl;  // 8 = 1(char c) + 3(padding) + 4(int i)
}
