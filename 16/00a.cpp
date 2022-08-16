#include <iostream>
#include <string>
#include <map>

using namespace std;

int main() {
    map<string, int> items = {
        {"hello", 1},
        {"world", 2},
    };

    int time = items["hello"];
    cout << time << endl;
}
