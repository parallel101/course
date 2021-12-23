#include <iostream>
#include <cstdlib>
#include <string>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

template <class T>
std::string cpp_type_name() {
    const char *name = typeid(T).name();
#if defined(__GNUC__) || defined(__clang__)
    int status;
    char *p = abi::__cxa_demangle(name, 0, 0, &status);
    std::string s = p;
    std::free(p);
#else
    std::string s = name;
#endif
    if (std::is_const_v<std::remove_reference_t<T>>)
        s += " const";
    if (std::is_volatile_v<std::remove_reference_t<T>>)
        s += " volatile";
    if (std::is_lvalue_reference_v<T>)
        s += " &";
    if (std::is_rvalue_reference_v<T>)
        s += " &&";
    return s;
}

#define SHOW(T) std::cout << cpp_type_name<T>() << std::endl;

int main() {
    int a, *p;
    SHOW(decltype(3.14f + a));
    SHOW(decltype(42));
    SHOW(decltype(&a));
    SHOW(decltype(p[0]));
    SHOW(decltype('a'));

    SHOW(decltype(a));    // int
    SHOW(decltype((a)));  // int &
    // 后者由于额外套了层括号，所以变成了 decltype(表达式)
}
