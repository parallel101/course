#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
using namespace std;

int main() {
    map<string, string> msg = {
        {"hello", "world"},
        {"fuck", "rust"},
    };
    print(msg);
    if (msg.count("fuck")) {
        print("存在fuck，其值为", msg.at("fuck"));
    } else {
        print("找不到fuck");
    }
    if (msg.count("suck")) {
        print("存在suck，其值为", msg.at("suck"));
    } else {
        print("找不到suck");
    }
    return 0;
}
