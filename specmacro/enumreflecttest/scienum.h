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
    typedef void type;
};

template <int Beg, int End, class F>
typename my_enable_if<Beg == End>::type static_for(F const &func) {
}

template <int Beg, int End, class F>
typename my_enable_if<Beg != End>::type static_for(F const &func) {
    func.template call<Beg>();
    static_for<Beg + 1, End>(func);
}

template <class T>
struct get_enum_name_functor {
    int n;
    std::string &s;

    get_enum_name_functor(int n, std::string &s) : n(n), s(s) {}

    template <int I>
    void call() const {
        if (n == I) s = details::get_enum_name_static<T, (T)I>();
    }
};

}

template <class T, T Beg, T End>
std::string get_enum_name(T n) {
    std::string s;
    details::static_for<Beg, End + 1>(details::get_enum_name_functor<T>(n, s));
    if (s.empty())
        return "";
#if defined(_MSC_VER)
    size_t pos = s.find(',');
    pos += 1;
    size_t pos2 = s.find('>', pos);
#else
    size_t pos = s.find("N = ");
    pos += 4;
    size_t pos2 = s.find_first_of(";]", pos);
#endif
    s = s.substr(pos, pos2 - pos);
    size_t pos3 = s.find("::");
    if (pos3 != s.npos)
        s = s.substr(pos3 + 2);
    return s;
}

template <class T>
std::string get_enum_name(T n) {
    return get_enum_name<T, (T)0, (T)256>(n);
}

template <class T, T Beg, T End>
T enum_from_name(std::string const &s) {
    for (int i = (int)Beg; i < (int)End; i++) {
        if (s == get_enum_name((T)i)) {
            return (T)i;
        }
    }
    throw;
}

template <class T>
T enum_from_name(std::string const &s) {
    return enum_from_name<T, (T)0, (T)256>(s);
}

}
