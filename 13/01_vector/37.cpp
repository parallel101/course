#include <vector>
#include <iostream>
#include <cstring>
#include "printer.h"
#include "mallochook.h"
using namespace std;

int main() {
    vector<int> a = {1, 2, 3, 4};
    cout << "before clear, capacity=" << a.capacity() << endl;
    a.clear();
    a.shrink_to_fit();
    cout << "after clear, capacity=" << a.capacity() << endl;
    return 0;
}
