#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ScopeProfiler.h"
using namespace std;

int main() {
    struct MyData {
        int value;
        explicit MyData(int value_) : value(value_) {}
    };
    map<string, unique_ptr<MyData>> m;
    m.insert({"answer", make_unique<MyData>(42)});
    m.insert({"fuck", make_unique<MyData>(985)});
    print(m.at("answer")->value);  // 42
    return 0;
}
