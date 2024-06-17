#include <bit>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

namespace reflect {

namespace details_ {

template <typename T>
struct any_except_ {
    template <typename U>
    constexpr operator U(); /* no body */

    constexpr operator T() = delete;
};

template <typename T, std::size_t ...Is>
static constexpr decltype((void)T{(Is, any_except_<T>{})...}) size__(std::index_sequence<Is...>); /* no body */

template <typename T, std::size_t I>
static constexpr decltype(size__<T>(std::make_index_sequence<I>{}), std::size_t{}) size_(int) {
    return I;
}

template <typename, std::size_t I>
static constexpr std::enable_if_t<I == 0, std::size_t> size_(...) {
    return 0;
}

template <typename T, std::size_t I>
static constexpr std::enable_if_t<I != 0, std::size_t> size_(...) {
    return size_<T, I - 1>(0);
}

template <std::size_t I>
struct tag_ {
    explicit tag_() = default;
};

#define SUCCESSOR_1 _0
#define SUCCESSOR_2 SUCCESSOR_1, _1
#define SUCCESSOR_3 SUCCESSOR_2, _2
#define SUCCESSOR_4 SUCCESSOR_3, _3
#define SUCCESSOR_5 SUCCESSOR_4, _4
#define SUCCESSOR_6 SUCCESSOR_5, _5
#define SUCCESSOR_7 SUCCESSOR_6, _6
#define SUCCESSOR_8 SUCCESSOR_7, _7
#define SUCCESSOR_9 SUCCESSOR_8, _8
#define SUCCESSOR_10 SUCCESSOR_9, _9
#define SUCCESSOR_11 SUCCESSOR_10, _10
#define SUCCESSOR_12 SUCCESSOR_11, _11
#define SUCCESSOR_13 SUCCESSOR_12, _12
#define SUCCESSOR_14 SUCCESSOR_13, _13
#define SUCCESSOR_15 SUCCESSOR_14, _14
#define SUCCESSOR_16 SUCCESSOR_15, _15
#define SUCCESSOR_17 SUCCESSOR_16, _16
#define SUCCESSOR_18 SUCCESSOR_17, _17
#define SUCCESSOR_19 SUCCESSOR_18, _18
#define SUCCESSOR_20 SUCCESSOR_19, _19
#define SUCCESSOR_21 SUCCESSOR_20, _20
#define SUCCESSOR_22 SUCCESSOR_21, _21
#define SUCCESSOR_23 SUCCESSOR_22, _22
#define SUCCESSOR_24 SUCCESSOR_23, _23
#define SUCCESSOR_25 SUCCESSOR_24, _24
#define SUCCESSOR_26 SUCCESSOR_25, _25
#define SUCCESSOR_27 SUCCESSOR_26, _26
#define SUCCESSOR_28 SUCCESSOR_27, _27
#define SUCCESSOR_29 SUCCESSOR_28, _28
#define SUCCESSOR_30 SUCCESSOR_29, _29
#define SUCCESSOR_31 SUCCESSOR_30, _30
#define SUCCESSOR_32 SUCCESSOR_31, _31

#define MEMBERS__(n, ...) \
    template <typename T, typename Visit> \
    constexpr auto members_(tag_<n>, T &&t, Visit &&visit) { \
        auto &&[__VA_ARGS__] = t; \
        return visit(__VA_ARGS__); \
    }

#define MEMBERS_(n, ...) MEMBERS__(n, __VA_ARGS__)
#define MEMBERS(n) MEMBERS_(n, SUCCESSOR_##n)

MEMBERS(1)
MEMBERS(2)
MEMBERS(3)
MEMBERS(4)
MEMBERS(5)
MEMBERS(6)
MEMBERS(7)
MEMBERS(8)
MEMBERS(9)
MEMBERS(10)
MEMBERS(11)
MEMBERS(12)
MEMBERS(13)
MEMBERS(14)
MEMBERS(15)
MEMBERS(16)
MEMBERS(17)
MEMBERS(18)
MEMBERS(19)
MEMBERS(20)
MEMBERS(21)
MEMBERS(22)
MEMBERS(23)
MEMBERS(24)
MEMBERS(25)
MEMBERS(26)
MEMBERS(27)
MEMBERS(28)
MEMBERS(29)
MEMBERS(30)
MEMBERS(31)
MEMBERS(32)

#undef MEMBERS__
#undef MEMBERS_
#undef MEMBERS
#undef SUCCESSOR_1
#undef SUCCESSOR_2
#undef SUCCESSOR_3
#undef SUCCESSOR_4
#undef SUCCESSOR_5
#undef SUCCESSOR_6
#undef SUCCESSOR_7
#undef SUCCESSOR_8
#undef SUCCESSOR_9
#undef SUCCESSOR_10
#undef SUCCESSOR_11
#undef SUCCESSOR_12
#undef SUCCESSOR_13
#undef SUCCESSOR_14
#undef SUCCESSOR_15
#undef SUCCESSOR_16
#undef SUCCESSOR_17
#undef SUCCESSOR_18
#undef SUCCESSOR_19
#undef SUCCESSOR_20
#undef SUCCESSOR_21
#undef SUCCESSOR_22
#undef SUCCESSOR_23
#undef SUCCESSOR_24
#undef SUCCESSOR_25
#undef SUCCESSOR_26
#undef SUCCESSOR_27
#undef SUCCESSOR_28
#undef SUCCESSOR_29
#undef SUCCESSOR_30
#undef SUCCESSOR_31
#undef SUCCESSOR_32

struct tuple_ref_ {
    template <typename ...Ts>
    constexpr std::tuple<Ts &...> operator()(Ts &...members) const {
        return {members...};
    }
};

struct tuple_move_ {
    template <typename ...Ts>
    constexpr std::tuple<std::remove_const_t<Ts>...> operator()(Ts &...members) const {
        return {std::move(members)...};
    }
};

struct tuple_copy_ {
    template <typename ...Ts>
    constexpr std::tuple<std::remove_const_t<Ts>...> operator()(Ts &...members) const {
        return {members...};
    }
};

struct tuple_types_ {
    template <typename ...Ts>
#if __cpp_lib_type_identity
    constexpr std::tuple<std::type_identity<std::remove_const_t<Ts>>...> operator()(Ts &...) const {
        return {};
    }
#else
    constexpr std::tuple<std::remove_const<Ts>...> operator()(Ts &...) const {
        return {};
    }
#endif
};

}

template <typename T>
constexpr std::size_t members_count() {
    static_assert(std::is_aggregate_v<T>);
    constexpr std::size_t size = details_::size_<T, 32>(0);
    static_assert(size > 0);
    return size;
}

template <typename T, typename Fn>
constexpr auto members_apply(T &&t, Fn &&fn) {
    return details_::members_(details_::tag_<members_count<std::decay_t<T>>()>{}, t, fn);
}

template <typename T, typename Each>
constexpr auto members_foreach(T &&t, Each &&each) {
    return members_apply(t, [&each] (auto &&...args) {
        return (each(args), ...);
    });
}

template <typename To, typename T, typename Each>
constexpr auto members_transform_to(T &&t, Each &&each) {
    return members_apply(t, [&each] (auto &&...args) {
        return To{each(args)...};
    });
}

template <typename T, typename Each>
constexpr auto members_transform(T &&t, Each &&each) {
    return members_apply(t, [&each] (auto &&...args) {
        return std::tuple{each(args)...};
    });
}

template <typename T>
constexpr auto members_reference(T &&t) {
    return members_apply(t, details_::tuple_ref_{});
}

template <typename T>
constexpr auto members_value(T &&t) {
    return members_apply(t, details_::tuple_move_{});
}

template <typename T, typename Each>
constexpr auto members_type() {
    return members_apply(std::declval<T &>(), details_::tuple_types_{});
}

template <char... chars>
struct static_string {
    static constexpr const char value[]{chars..., 0};

    static constexpr std::size_t size() noexcept {
        return sizeof...(chars);
    }

    static constexpr const char *data() noexcept {
        return value;
    }

    constexpr operator const char *() const noexcept {
        return value;
    }
};

template <class T>
struct member_names_trait {
    using type = static_string<>;
};

namespace details_ {

template <char c, char ...chars>
constexpr static_string<c, chars...> string_prepend_(static_string<chars...>);

template <char c, char ...chars>
constexpr static_string<chars..., c> string_append_(static_string<chars...>);

template <std::size_t N, const char *str, std::size_t I = 0>
struct string_builder_ {
    using type = decltype(string_prepend_<str[I]>(typename string_builder_<N, str, I + 1>::type{}));
};

template <std::size_t N, const char *str>
struct string_builder_<N, str, N> {
    using type = static_string<>;
};

constexpr std::size_t static_string_length_(const char *str) {
    for (std::size_t i = 0;; ++i) {
        if (str[i] == '\0') return i;
    }
}

template <std::size_t I, bool = true>
constexpr auto index_to_string_() {
    return string_append_(I % 10 + '0', details_::index_to_string_<I / 10, false>());
}

template <>
constexpr auto index_to_string_<0, false>() {
    return static_string<>();
}

template <>
constexpr auto index_to_string_<0, true>() {
    return static_string<'0'>();
}

template <std::size_t, class>
struct slice_string_at_comma_;

template <class>
struct stop_string_at_comma_;

template <>
struct stop_string_at_comma_<static_string<>> {
    using type = static_string<>;
};

template <char ...chars>
struct stop_string_at_comma_<static_string<' ', chars...>>
    : stop_string_at_comma_<static_string<chars...>> {
};

template <char ...chars>
struct stop_string_at_comma_<static_string<',', chars...>> {
    using type = static_string<>;
};

template <char c, char ...chars>
struct stop_string_at_comma_<static_string<c, chars...>> {
    using type = decltype(string_prepend_<c>(typename stop_string_at_comma_<static_string<chars...>>::type{}));
};

template <std::size_t I, char c, char ...chars>
struct slice_string_at_comma_<I, static_string<c, chars...>>
    : slice_string_at_comma_<I - (c == ','), static_string<chars...>> {};

template <char c, char ...chars>
struct slice_string_at_comma_<0, static_string<c, chars...>>
    : stop_string_at_comma_<static_string<c, chars...>> {};

template <typename T, std::size_t ...Is>
constexpr auto members_name_(std::index_sequence<Is...>) {
    using str = typename member_names_trait<T>::type;
    return std::tuple{typename slice_string_at_comma_<Is, str>::type{}...};
}

}

template <const char *str>
using make_static_string = typename details_::string_builder_<details_::static_string_length_(str), str>::type;

#define DEFINE_STATIC_STRING(...) \
{ \
    static constexpr const char _static_buffer[] = __VA_ARGS__; \
    using type = reflect::make_static_string<_static_buffer>; \
}

#define DEFINE_MEMBER_NAMES(type, ...) \
template <> \
struct reflect::member_names_trait<type> DEFINE_STATIC_STRING(#__VA_ARGS__);

template <typename T>
constexpr auto members_name() {
    constexpr std::size_t size = members_count<T>();
    return details_::members_name_<T>(std::make_index_sequence<size>{});
}

}

#include <string>
#include <iostream>

struct X {
    std::string s;
};

struct Y {
    double a, b, c;
};

struct Z {
    int f1;
    long f2;
    size_t f3;
    std::string f4;
};

DEFINE_MEMBER_NAMES(Y, a, b, c);

int main() {
    std::cout << std::get<1>(reflect::members_name<Y>()).value << '\n';
    std::cout << reflect::members_count<X>() << '\n';
    std::cout << reflect::members_count<Y>() << '\n';
    std::cout << reflect::members_count<Z>() << '\n';
    std::cout << std::get<3>(reflect::members_reference(Z{5, 6, 7, "8h"})) << '\n';
}
