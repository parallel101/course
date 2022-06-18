#include <set>
#include <vector>
#include <unordered_set>
#include "printer.h"

using namespace std;

int main() {
    vector<int> arr(100);

    auto comp = [&] (int i, int j) {
        return arr[i] < arr[j];
    };

    set<int, decltype(comp)> a({1, 4, 2, 8, 5, 7}, comp);
    unordered_set<int> b = {1, 4, 2, 8, 5, 7};

    cout << "set: " << a << endl;
    cout << "unordered_set: " << b << endl;

    return 0;
}
