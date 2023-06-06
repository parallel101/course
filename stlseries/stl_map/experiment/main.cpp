#include "bits_stdc++.h"
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
#include "hash.h"
using namespace std;

int main() {
    unordered_map<string, int, generic_hash<>, std::equal_to<>> table;
    pair<const string, int> entry = {"answer", 42};
    table.insert(std::as_const(entry));
    string_view key = "hello";
    bool res = std::equal_to<>()(key, entry.first);
    table.find(key);
    string s = "";
    return 0;
}
