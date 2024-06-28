/* // The template we want to pass a string to */
/* template <int... Args> */
/* struct foo { */
/*   // It needs one helper function for decltype magic, this could be avoided though */
/*   template <int N> */
/*   static foo<N, Args...>  add_one(); */
/* }; */
/*  */
/* // This is the string we want to use with foo, simulating foo<"Hello world!" __FILE__>: */
/* constexpr const char *teststr = "Hello world!" __FILE__; */
/*  */
/* // Get char N of a string literal */
/* constexpr int strchr(const char *str, int N) { return str[N]; } */
/*  */
/* // recursive helper to build the typedef from teststr */
/* template <int N, int P=0> */
/* struct builder { */
/*    typedef typename builder<N, P+1>::type child; */
/*    typedef decltype(child::template add_one<strchr(teststr,P)>()) type; */
/* }; */
/*  */
/* template <int N> */
/* struct builder<N,N> { */
/*   typedef foo<strchr(teststr, N)> type; */
/* }; */
/*  */
/* // compile time strlen */
/* constexpr int slen(const char *str) { */
/*   return *str ? 1 + slen(str+1) : 0; */
/* } */
/*  */
/* int main() { */
/*   builder<slen(teststr)>::type test; */
/*   // compile error to force the type to be printed: */
/*   int foo = test; */
/* } */

#include <utility>

template <char... chars>
struct static_string {
    static constexpr std::size_t size() noexcept {
        return sizeof...(chars);
    }
};

template <char c, char ...chars>
constexpr static_string<c, chars...> string_prepend(static_string<chars...>);

template <std::size_t N, const char *str, std::size_t I = 0>
struct string_builder {
    using type = decltype(string_prepend<str[I]>(typename string_builder<N, str, I + 1>::type{}));
};

template <std::size_t N, const char *str>
struct string_builder<N, str, N> {
    using type = static_string<>;
};

constexpr std::size_t static_string_length(const char *str) {
    for (std::size_t i = 0;; ++i) {
        if (str[i] == '\0') return i;
    }
}

template <const char *str>
using make_static_string = typename string_builder<static_string_length(str), str>::type;

int main() {
    static constexpr const char s[] = "hello";
    auto m = make_static_string<s>();
    m.size();
}
