#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
using namespace std;

int main() {
    map<string, int> m = {
        {"answer", 42},
        {"timeout", 4096},
    };
    m.insert({
        {"timeout", 985},
        {"delay", 211},
    });
    print(m);
}
