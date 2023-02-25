#include <bits/stdc++.h>
#include "print.h"
using namespace std;

int main() {
const map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config["timeout"]);  // 编译出错
}
