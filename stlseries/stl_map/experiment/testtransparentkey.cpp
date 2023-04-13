#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

struct FatKey   { int x; int data[1000]; };
struct LightKey { int x; };
// Note: as detailed above, the container must use std::less<> (or other 
//   transparent Comparator) to access these overloads.
// This includes standard overloads, such as between std::string and std::string_view.
bool operator<(const FatKey& fk, const LightKey& lk) { return fk.x < lk.x; }
bool operator<(const LightKey& lk, const FatKey& fk) { return lk.x < fk.x; }
bool operator<(const FatKey& fk1, const FatKey& fk2) { return fk1.x < fk2.x; }

int main() {
    // transparent comparison demo
    std::map<FatKey, char, std::less<>> example = {{{1, {}}, 'a'}, {{2, {}}, 'b'}};
 
    LightKey lk = {2};
    if (auto search = example.find(lk); search != example.end())
        std::cout << "Found " << search->first.x << " " << search->second << '\n';
    else
        std::cout << "Not found\n";
}
