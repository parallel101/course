#pragma once

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

#define REFLECT_SUCC_1 _0
#define REFLECT_SUCC_2 REFLECT_SUCC_1, _1
#define REFLECT_SUCC_3 REFLECT_SUCC_2, _2
#define REFLECT_SUCC_4 REFLECT_SUCC_3, _3
#define REFLECT_SUCC_5 REFLECT_SUCC_4, _4
#define REFLECT_SUCC_6 REFLECT_SUCC_5, _5
#define REFLECT_SUCC_7 REFLECT_SUCC_6, _6
#define REFLECT_SUCC_8 REFLECT_SUCC_7, _7
#define REFLECT_SUCC_9 REFLECT_SUCC_8, _8
#define REFLECT_SUCC_10 REFLECT_SUCC_9, _9
#define REFLECT_SUCC_11 REFLECT_SUCC_10, _10
#define REFLECT_SUCC_12 REFLECT_SUCC_11, _11
#define REFLECT_SUCC_13 REFLECT_SUCC_12, _12
#define REFLECT_SUCC_14 REFLECT_SUCC_13, _13
#define REFLECT_SUCC_15 REFLECT_SUCC_14, _14
#define REFLECT_SUCC_16 REFLECT_SUCC_15, _15
#define REFLECT_SUCC_17 REFLECT_SUCC_16, _16
#define REFLECT_SUCC_18 REFLECT_SUCC_17, _17
#define REFLECT_SUCC_19 REFLECT_SUCC_18, _18
#define REFLECT_SUCC_20 REFLECT_SUCC_19, _19
#define REFLECT_SUCC_21 REFLECT_SUCC_20, _20
#define REFLECT_SUCC_22 REFLECT_SUCC_21, _21
#define REFLECT_SUCC_23 REFLECT_SUCC_22, _22
#define REFLECT_SUCC_24 REFLECT_SUCC_23, _23
#define REFLECT_SUCC_25 REFLECT_SUCC_24, _24
#define REFLECT_SUCC_26 REFLECT_SUCC_25, _25
#define REFLECT_SUCC_27 REFLECT_SUCC_26, _26
#define REFLECT_SUCC_28 REFLECT_SUCC_27, _27
#define REFLECT_SUCC_29 REFLECT_SUCC_28, _28
#define REFLECT_SUCC_30 REFLECT_SUCC_29, _29
#define REFLECT_SUCC_31 REFLECT_SUCC_30, _30
#define REFLECT_SUCC_32 REFLECT_SUCC_31, _31

#define REFLECT_MEMBERS__(n, ...) \
    template <typename T, typename Visit> \
    constexpr auto members_(tag_<n>, T &&t, Visit &&visit) { \
        auto &&[__VA_ARGS__] = t; \
        return visit(__VA_ARGS__); \
    }

#define REFLECT_MEMBERS_(n, ...) REFLECT_MEMBERS__(n, __VA_ARGS__)
#define REFLECT_MEMBERS(n) REFLECT_MEMBERS_(n, REFLECT_SUCC_##n)

REFLECT_MEMBERS(1)
REFLECT_MEMBERS(2)
REFLECT_MEMBERS(3)
REFLECT_MEMBERS(4)
REFLECT_MEMBERS(5)
REFLECT_MEMBERS(6)
REFLECT_MEMBERS(7)
REFLECT_MEMBERS(8)
REFLECT_MEMBERS(9)
REFLECT_MEMBERS(10)
REFLECT_MEMBERS(11)
REFLECT_MEMBERS(12)
REFLECT_MEMBERS(13)
REFLECT_MEMBERS(14)
REFLECT_MEMBERS(15)
REFLECT_MEMBERS(16)
REFLECT_MEMBERS(17)
REFLECT_MEMBERS(18)
REFLECT_MEMBERS(19)
REFLECT_MEMBERS(20)
REFLECT_MEMBERS(21)
REFLECT_MEMBERS(22)
REFLECT_MEMBERS(23)
REFLECT_MEMBERS(24)
REFLECT_MEMBERS(25)
REFLECT_MEMBERS(26)
REFLECT_MEMBERS(27)
REFLECT_MEMBERS(28)
REFLECT_MEMBERS(29)
REFLECT_MEMBERS(30)
REFLECT_MEMBERS(31)
REFLECT_MEMBERS(32)

#undef REFLECT_MEMBERS__
#undef REFLECT_MEMBERS_
#undef REFLECT_MEMBERS
#undef REFLECT_SUCC_1
#undef REFLECT_SUCC_2
#undef REFLECT_SUCC_3
#undef REFLECT_SUCC_4
#undef REFLECT_SUCC_5
#undef REFLECT_SUCC_6
#undef REFLECT_SUCC_7
#undef REFLECT_SUCC_8
#undef REFLECT_SUCC_9
#undef REFLECT_SUCC_10
#undef REFLECT_SUCC_11
#undef REFLECT_SUCC_12
#undef REFLECT_SUCC_13
#undef REFLECT_SUCC_14
#undef REFLECT_SUCC_15
#undef REFLECT_SUCC_16
#undef REFLECT_SUCC_17
#undef REFLECT_SUCC_18
#undef REFLECT_SUCC_19
#undef REFLECT_SUCC_20
#undef REFLECT_SUCC_21
#undef REFLECT_SUCC_22
#undef REFLECT_SUCC_23
#undef REFLECT_SUCC_24
#undef REFLECT_SUCC_25
#undef REFLECT_SUCC_26
#undef REFLECT_SUCC_27
#undef REFLECT_SUCC_28
#undef REFLECT_SUCC_29
#undef REFLECT_SUCC_30
#undef REFLECT_SUCC_31
#undef REFLECT_SUCC_32

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
constexpr std::size_t count() {
    static_assert(std::is_aggregate_v<T>);
    constexpr std::size_t size = details_::size_<T, 32>(0);
    static_assert(size > 0);
    return size;
}

template <typename T, typename Fn>
constexpr auto apply(T &&t, Fn &&fn) {
    return details_::members_(details_::tag_<count<std::decay_t<T>>()>{}, t, fn);
}

template <typename T, typename Each>
constexpr auto foreach(T &&t, Each &&each) {
    return reflect::apply(t, [&each] (auto &&...args) {
        return (each(args), ...);
    });
}

template <typename T>
constexpr auto reference(T &&t) {
    return reflect::apply(t, details_::tuple_ref_{});
}

template <typename T>
constexpr auto value(T &&t) {
    return reflect::apply(t, details_::tuple_move_{});
}

template <typename T, typename Each>
constexpr auto type() {
    return reflect::apply(std::declval<T &>(), details_::tuple_types_{});
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
constexpr auto name_(std::index_sequence<Is...>) {
    using str = typename member_names_trait<T>::type;
    return std::tuple{typename slice_string_at_comma_<Is, str>::type{}...};
}

}

template <typename T>
constexpr auto name() {
    constexpr std::size_t size = reflect::count<T>();
    return details_::name_<T>(std::make_index_sequence<size>{});
}

template <const char *str>
using make_static_string = typename details_::string_builder_<details_::static_string_length_(str), str>::type;

#define REFLECT_STATIC_STRING(...) \
{ \
    static constexpr const char _static_buffer[] = __VA_ARGS__; \
    using type = reflect::make_static_string<_static_buffer>; \
}

#define REFLECT_MEMBERS(type, ...) \
template <> \
struct reflect::member_names_trait<type> REFLECT_STATIC_STRING(#__VA_ARGS__);

}
