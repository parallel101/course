#pragma once

#include <type_traits>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <iomanip>
#include <string>
#include <string_view>
#include <optional>
#include <variant>

namespace _print_details {

template <class T, class = void>
struct _printer {
    static void print(std::ostream &os, T const &t) {
        os << t;
    }

    using is_default_print = std::true_type;
};

template <class T, class = void>
struct _is_default_printable : std::false_type {
};

template <class T>
struct _is_default_printable<T, std::void_t<std::pair<typename _printer<T>::is_default_print, decltype(std::declval<std::ostream &>() << std::declval<T const &>())>>> : std::true_type {
};

template <class T, class = void>
struct _is_printer_printable : std::true_type {
};

template <class T>
struct _is_printer_printable<T, std::void_t<typename _printer<T>::is_default_print>> : std::false_type {
};

template <class T, class = void>
struct is_printable : std::disjunction<_is_default_printable<T>, _is_printer_printable<T>> {
};

template <class T>
using _rmcvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T, class U = void, class = void>
struct _enable_if_tuple {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_tuple<T, U, std::void_t<decltype(std::tuple_size<T>::value)>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_glmvec {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_glmvec<T, U, std::enable_if_t<
        std::is_same_v<decltype(T()[0] * T()), T> &&
        std::is_same_v<decltype(T() * T()[0]), T> &&
        std::is_same_v<decltype(T().x + T()), T> &&
        std::is_same_v<decltype(T().x), typename T::value_type> &&
        std::is_same_v<decltype(T()[0]), typename T::value_type &> &&
        std::is_same_v<decltype(T() == T()), bool> &&
        std::is_same_v<decltype(T(T()[0])), T> &&
        std::is_same_v<decltype(T().length()), typename T::length_type> &&
        std::is_same_v<decltype(std::declval<typename T::bool_type>().x), bool>
>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_glmmat {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_glmmat<T, U, typename _enable_if_glmvec<typename T::col_type,
        typename _enable_if_glmvec<typename T::row_type, std::enable_if_t<
        std::is_same_v<decltype(T()[0] * T()), typename T::row_type> &&
        std::is_same_v<decltype(T() * T()[0]), typename T::col_type> &&
        std::is_same_v<decltype(T()[0][0] * T()), T> &&
        std::is_same_v<decltype(T()[0][0]), typename T::value_type &>
>>::type>::type> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_map {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_map<T, U, std::enable_if_t<std::is_same_v<typename T::value_type, std::pair<typename T::key_type const, typename T::mapped_type>>>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_iterable {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_iterable<T, U, std::void_t<typename std::iterator_traits<decltype(std::begin(std::declval<T const &>()))>::value_type>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_has_print {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_has_print<T, U, std::void_t<decltype(std::declval<T const &>().do_print(std::declval<std::ostream &>()))>> {
    using type = U;
};

template <class T>
struct _is_char : std::false_type {
};

template <>
struct _is_char<char> : std::true_type {
};

template <>
struct _is_char<wchar_t> : std::true_type {
};

template <class T, class U = void, class = void>
struct _enable_if_char {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_char<T, U, std::enable_if_t<_is_char<T>::value>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_string {
    using not_type = U;
};

template <class T, class Alloc, class Traits, class U>
struct _enable_if_string<std::basic_string<T, Alloc, Traits>, U, typename _enable_if_char<T>::type> {
    using type = U;
};

template <class T, class Traits, class U>
struct _enable_if_string<std::basic_string_view<T, Traits>, U, typename _enable_if_char<T>::type> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_c_str {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_c_str<T, U, std::enable_if_t<std::is_pointer_v<std::decay_t<T>> && _is_char<std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>>::value>> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_optional {
    using not_type = U;
};

template <class T, class U>
struct _enable_if_optional<std::optional<T>, U, void> {
    using type = U;
};

template <class T, class U = void, class = void>
struct _enable_if_variant {
    using not_type = U;
};

template <class ...Ts, class U>
struct _enable_if_variant<std::variant<Ts...>, U, void> {
    using type = U;
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T>::type> {
    static void print(std::ostream &os, T const &t) {
        t.do_print(os);
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_iterable<T, typename _enable_if_c_str<T, typename _enable_if_string<T, typename _enable_if_map<T>::not_type>::not_type>::not_type>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        os << "{";
        bool once = false;
        for (auto const &v: t) {
            if (once) {
                os << ", ";
            } else {
                once = true;
            }
            _printer<_rmcvref_t<decltype(v)>>::print(os, v);
        }
        os << "}";
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_tuple<T, typename _enable_if_iterable<T>::not_type>::type>::not_type> {
    template <std::size_t ...Is>
    static void _unrolled_print(std::ostream &os, T const &t, std::index_sequence<Is...>) {
        os << "{";
        ((_printer<_rmcvref_t<std::tuple_element_t<Is, T>>>::print(os, std::get<Is>(t)), os << ", "), ...);
        if constexpr (sizeof...(Is) != 0) _printer<_rmcvref_t<std::tuple_element_t<sizeof...(Is), T>>>::print(os, std::get<sizeof...(Is)>(t));
        os << "}";
    }

    static void print(std::ostream &os, T const &t) {
        _unrolled_print(os, t, std::make_index_sequence<std::max(static_cast<std::size_t>(1), std::tuple_size_v<T>) - 1>{});
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_glmvec<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        os << "{";
        bool once = false;
        for (typename T::length_type i = 0; i < t.length(); i++) {
            if (once) {
                os << ", ";
            } else {
                once = true;
            }
            _printer<_rmcvref_t<typename T::value_type>>::print(os, t[i]);
        }
        os << "}";
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_glmmat<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        os << "{";
        bool once = false;
        for (typename T::length_type i = 0; i < t.length(); i++) {
            if (once) {
                os << ",\n ";
            } else {
                once = true;
            }
            _printer<_rmcvref_t<typename T::col_type>>::print(os, t[i]);
        }
        os << "}";
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_map<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        os << "{";
        bool once = false;
        for (auto const &[k, v]: t) {
            if (once) {
                os << ", ";
            } else {
                once = true;
            }
            _printer<_rmcvref_t<decltype(k)>>::print(os, k);
            os << ": ";
            _printer<_rmcvref_t<decltype(v)>>::print(os, v);
        }
        os << "}";
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_string<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        os << std::quoted(t);
    }
};

template <class T>
struct _printer<T, typename _enable_if_c_str<T>::type> {
    static void print(std::ostream &os, T const &t) {
        os << t;
    }
};

template <class T>
struct _printer<T, typename _enable_if_char<T>::type> {
    static void print(std::ostream &os, T const &t) {
        T s[2] = {t, T('\0')};
        os << std::quoted(s, T('\''));
    }
};

template <>
struct _printer<std::nullptr_t, void> {
    static void print(std::ostream &os, std::nullptr_t const &) {
        os << "nullptr";
    }
};

template <>
struct _printer<std::nullopt_t, void> {
    static void print(std::ostream &os, std::nullopt_t const &) {
        os << "nullopt";
    }
};

template <>
struct _printer<std::monostate, void> {
    static void print(std::ostream &os, std::monostate const &) {
        os << "monostate";
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_optional<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        if (t.has_value()) {
            _printer<typename T::value_type>::print(os, *t);
        } else {
            _printer<std::nullopt_t>::print(os, std::nullopt);
        }
    }
};

template <class T>
struct _printer<T, typename _enable_if_has_print<T, typename _enable_if_variant<T>::type>::not_type> {
    static void print(std::ostream &os, T const &t) {
        std::visit([&] (auto const &v) {
            _printer<_rmcvref_t<decltype(v)>>::print(os, v);
        }, t);
    }
};

template <>
struct _printer<bool, void> {
    static void print(std::ostream &os, bool const &t) {
        if (t) {
            os << "true";
        } else {
            os << "false";
        }
    }
};

template <class T0, class ...Ts>
void fprint(std::ostream &os, T0 const &t0, Ts const &...ts) {
    _printer<_rmcvref_t<T0>>::print(os, t0);
    ((os << " ", _printer<_rmcvref_t<Ts>>::print(os, ts)), ...);
    os << "\n";
}

template <class T0, class ...Ts>
void fprintnl(std::ostream &os, T0 const &t0, Ts const &...ts) {
    _printer<_rmcvref_t<T0>>::print(os, t0);
    (_printer<_rmcvref_t<Ts>>::print(os, ts), ...);
}

template <class T0, class ...Ts>
void print(T0 const &t0, Ts const &...ts) {
    fprint(std::cout, t0, ts...);
}

template <class T0, class ...Ts>
void printnl(T0 const &t0, Ts const &...ts) {
    fprintnl(std::cout, t0, ts...);
}

template <class T0, class ...Ts>
void eprint(T0 const &t0, Ts const &...ts) {
    fprint(std::cerr, t0, ts...);
}

template <class T0, class ...Ts>
void eprintnl(T0 const &t0, Ts const &...ts) {
    fprintnl(std::cerr, t0, ts...);
}

template <class T0, class ...Ts>
std::string to_string(T0 const &t0, Ts const &...ts) {
    std::ostringstream oss;
    fprintnl(oss, t0, ts...);
    return oss.str();
}

template <class T, class = void>
class print_adaptor {
    T const &t;

public:
    explicit constexpr print_adaptor(T const &t_) : t(t_) {
    }

    friend std::ostream &operator<<(std::ostream &os, print_adaptor const &&self) {
        auto oldflags = os.flags();
        os << "[object 0x" << std::hex << reinterpret_cast<std::uintptr_t>(
            reinterpret_cast<void const volatile *>(std::addressof(self.t))) << ']';
        os.flags(oldflags);
        return os;
    }
};

template <class T>
class print_adaptor<T, std::enable_if_t<is_printable<T>::value>> {
    T const &t;

public:
    explicit constexpr print_adaptor(T const &t_) : t(t_) {
    }

    friend std::ostream &operator<<(std::ostream &os, print_adaptor const &&self) {
        fprintnl(os, self.t);
        return os;
    }
};

template <class T>
explicit print_adaptor(T const &) -> print_adaptor<T>;

template <class T>
struct as_hex {
    T m_value;

    explicit as_hex(T value) : m_value(std::move(value)) {}

    void do_print(std::ostream &os) const {
        auto tmp = os.flags();
        os << "0x" << std::hex << print_adaptor(m_value);
        os.flags(tmp);
    }
};

template <class T>
as_hex(T) -> as_hex<T>;

}

using _print_details::print;
using _print_details::printnl;
using _print_details::eprint;
using _print_details::eprintnl;
using _print_details::fprint;
using _print_details::fprintnl;
using _print_details::to_string;
using _print_details::print_adaptor;
using _print_details::is_printable;
using _print_details::as_hex;

// Usage:
//
// map<string, optional<int>> m = {{"hello", 42}, {"world", nullopt}};
// print(m);  // {"hello": 42, "world": nullopt}
