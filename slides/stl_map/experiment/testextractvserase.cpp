#include <algorithm>
#include <bits/stdc++.h>
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
using namespace std;

template <class K, class V, class Cond>
void filter_with_extract_with_hint(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    auto hint = tab2.begin();
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            auto next_it = it;
            ++next_it;
            auto node = tab1.extract(it);
            hint = tab2.insert(hint, std::move(node));
            it = next_it;
        } else {
            ++it;
        }
    }
}

template <class K, class V, class Cond>
void filter_with_erase_with_hint(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    auto hint = tab2.begin();
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            it = tab1.erase(it);
            auto kv = std::move(*it);
            hint = tab2.insert(hint, std::move(kv));
        } else {
            ++it;
        }
    }
}

template <class K, class V, class Cond>
void filter_with_extract(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            auto next_it = it;
            ++next_it;
            auto node = tab1.extract(it);
            tab2.insert(std::move(node));
            it = next_it;
        } else {
            ++it;
        }
    }
}

template <class K, class V, class Cond>
void filter_with_erase(map<K, V> &tab1, map<K, V> &tab2, Cond &&cond) {
    DefScopeProfiler;
    for (auto it = tab1.begin(); it != tab1.end(); ) {
        if (cond(it->first)) {
            it = tab1.erase(it);
            auto kv = std::move(*it);
            tab2.insert(std::move(kv));
        } else {
            ++it;
        }
    }
}

template <class K, class V>
map<K, V> generate_table() {
    map<K, V> tab;
    auto hint = tab.begin();
    for (int i = 0; i < 20000; i++) {
        hint = tab.try_emplace(hint, i, "entry" + std::to_string(i));
    }
    return tab;
}

int main() {
    for (int i = 0; i < 1000; i++) {
        auto tab1 = generate_table<int, string>();
        auto tab2 = map<int, string>();
        filter_with_extract(tab1, tab2, [] (int k) { return k % 2 == 0; });
        doNotOptimize(tab1);
        doNotOptimize(tab2);
    }
    for (int i = 0; i < 1000; i++) {
        auto tab1 = generate_table<int, string>();
        auto tab2 = map<int, string>();
        filter_with_erase(tab1, tab2, [] (int k) { return k % 2 == 0; });
        doNotOptimize(tab1);
        doNotOptimize(tab2);
    }
    for (int i = 0; i < 1000; i++) {
        auto tab1 = generate_table<int, string>();
        auto tab2 = map<int, string>();
        filter_with_extract_with_hint(tab1, tab2, [] (int k) { return k % 2 == 0; });
        doNotOptimize(tab1);
        doNotOptimize(tab2);
    }
    for (int i = 0; i < 1000; i++) {
        auto tab1 = generate_table<int, string>();
        auto tab2 = map<int, string>();
        filter_with_erase_with_hint(tab1, tab2, [] (int k) { return k % 2 == 0; });
        doNotOptimize(tab1);
        doNotOptimize(tab2);
    }
    printScopeProfiler();
}
