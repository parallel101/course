#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
using namespace std;

int main() {
    print("1", "2");
#if 0
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
if (msg.count("fuck")) {
    print((string)"存在fuck，其值为", msg.at("fuck"));
} else {
    print((string)"找不到fuck");
}
if (msg.count("suck")) {
    print((string)"存在suck，其值为", msg.at("suck"));
} else {
    print((string)"找不到suck");
}
#endif
    return 0;
}
