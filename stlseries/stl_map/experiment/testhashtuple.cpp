#include "bits_stdc++.h"
#include "print.h"
#include "cppdemangle.h"
#include "map_get.h"
#include "ppforeach.h"
#include "ScopeProfiler.h"
#include "hash.h"
using namespace std;

template <class ...Ts>
inline size_t _hash_tuple_impl(Ts const &...ts) {
    size_t h = 0;
    ((h ^= std::hash<Ts>()(ts) + 0x9e3779b9 + (h << 6) + (h >> 2)), ...);
    return h;
}

template <class ...Ts>
struct std::hash<std::tuple<Ts...>> {
    size_t operator()(std::tuple<Ts...> const &x) const {
        return std::apply(_hash_tuple_impl<Ts...>, x);
    }
};

int main() {
    tuple<int, int> t(42, 64);
    size_t h = hash<tuple<int, int>>()(t);
    print(t, "的哈希值是:", h);
    return 0;
}
