#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

int main() {
    map<int, string> hells = {
        {666, "devil"},
    };
    map<int, string> schools = {
        {985, "professor"},
        {211, "doctor"},
        {996, "fucker"},
    };
    auto node = schools.extract(996);
    hells.insert(std::move(node));
    print(schools);
    print(hells);
}
