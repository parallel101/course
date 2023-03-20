#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

auto generate_vector(size_t n) {
    vector<pair<int, int>> arr;
    arr.reserve(n);
    for (size_t i = 0; i < n; i++) {
        arr.push_back({i, 0});
    }
    return arr;
}

void test_bracket(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    for (auto const &[k, v]: arr) {
        tab[k] = v;
    }
}

void test_insert_or_assign(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    for (auto const &[k, v]: arr) {
        tab.insert_or_assign(k, v);
    }
}

void test_insert(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    for (auto const &[k, v]: arr) {
        tab.insert({k, v});
    }
}

void test_insert_with_hint(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    auto hint = tab.begin();
    for (auto const &[k, v]: arr) {
        hint = tab.insert(hint, {k, v});
    }
}

void test_try_emplace(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    for (auto const &[k, v]: arr) {
        tab.try_emplace(k, v);
    }
}

void test_try_emplace_with_hint(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    auto hint = tab.begin();
    for (auto const &[k, v]: arr) {
        hint = tab.try_emplace(hint, k, v);
    }
}

void test_insert_batched(
    const char *scope_name,
    vector<pair<int, int>> const &arr,
    map<int, int> &tab) {
    ScopeProfiler _{scope_name};
    tab.insert(arr.begin(), arr.end());
}

int main() {
    const int kTimes = 100;
    const int kElements = 50000;
    auto arr = generate_vector(kElements);
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_bracket("sorted []", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_or_assign("sorted insert_or_assign", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert("sorted insert", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_try_emplace("sorted try_emplace", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_with_hint("sorted insert with hint", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_try_emplace_with_hint("sorted try_emplace with hint", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_batched("sorted insert batched", arr, tab);
        doNotOptimize(tab);
    }
    shuffle(arr.begin(), arr.end(), mt19937{});
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_bracket("random []", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_or_assign("random insert_or_assign", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert("random insert", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_try_emplace("random try_emplace", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_with_hint("random insert with hint", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_try_emplace_with_hint("random try_emplace with hint", arr, tab);
        doNotOptimize(tab);
    }
    for (int i = 0; i < kTimes; i++) {
        map<int, int> tab;
        test_insert_batched("random insert batched", arr, tab);
        doNotOptimize(tab);
    }
    printScopeProfiler();
    return 0;
}
