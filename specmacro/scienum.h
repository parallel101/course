#pragma once

#include <string>

namespace scienum {

namespace details {

template <class T, T N>
const char *get_enum_name_static() {
#if defined(_MSC_VER)
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

template <bool Cond>
struct my_enable_if {
};

template <>
struct my_enable_if<true> {
    typedef int type;
};

template <int Beg, int End, class F, typename my_enable_if<Beg == End>::type = 0>
void static_for(F const &func) {
}

template <int Beg, int End, class F, typename my_enable_if<Beg != End>::type = 0>
void static_for(F const &func) {
    struct int_constant {
        enum { value = Beg };
    };
    func(int_constant());
    static_for<Beg + 1, End>(func);
}

}

template <int Beg = 0, int End = 256, class T>
std::string get_enum_name(T n) {
    std::string s;
    details::static_for<Beg, End + 1>([&] (auto i) {
        if (n == (T)i.value) s = details::get_enum_name_static<T, (T)i.value>();
    });
    if (s.empty())
        return std::to_string((int)n);
#if defined(_MSC_VER)
    auto pos = s.find(',');
    pos += 1;
    auto pos2 = s.find('>', pos);
#else
    auto pos = s.find("N = ");
    pos += 4;
    auto pos2 = s.find_first_of(";]", pos);
#endif
    s = s.substr(pos, pos2 - pos);
    auto pos3 = s.find("::");
    if (pos3 != s.npos)
        s = s.substr(pos3 + 2);
    return s;
}

template <class T, int Beg = 0, int End = 256>
T enum_from_name(std::string const &s) {
    for (int i = Beg; i < End; i++) {
        if (s == get_enum_name((T)i)) {
            return (T)i;
        }
    }
    throw;
}

}
