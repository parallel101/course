#include <map>
#include <string>
#include <iostream>

int main() {
    std::map<int, int> m = {
        {1, 2},
        {4, 3},
        {5, 4},
        {7, 5},
        {10, 6},
    };
    //for (auto &[k, v]: m) {
        //m.erase(k);
    //}
    //for (auto it = m.begin(); it != m.end(); ++it) {
        //auto &[k, v] = *it;
        //m.erase(k);
    //}
    for (auto it = m.begin(); it != m.end();) {
        auto &[k, v] = *it;
        ++it;
        m.erase(k);
    }
    return 0;
}
