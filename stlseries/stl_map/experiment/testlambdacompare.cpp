#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

int main() {
    auto cmp = [] (std::string const &lhs, std::string const &rhs) {
        return std::lexicographical_compare
            ( lhs.begin(), lhs.end()
            , rhs.begin(), rhs.end()
            , [] (char lhs, char rhs) {
                return std::toupper(lhs) < std::toupper(rhs);
            });
    };
    map<string, string, decltype(cmp)> m(cmp);
    m = {
        {{"Fuck"}, "rust"},
        {{"fUCK"}, "java"},
        {{"Study"}, "cpp"},
    };
    print(m);
    auto val = m.at({"fuck"});
    print(val);
    m.value_comp()({"hello", "hello"}, {"hello", "hello"});
    return 0;
}
