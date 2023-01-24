#include <iostream>
#include <string>
#include <map>

using namespace std;

int main() {
    map<string, int> items = {
        {"hello", 1},
        {"world", 2},
    };

    items["time"] = 4;
    int time = items.at("time");
    cout << time << endl;
}
