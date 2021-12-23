#include <cstdio>

int t;

int const &func_ref() {
    return t;
}

int const &func_cref() {
    return t;
}

int func_val() {
    return t;
}

int main() {
    decltype(auto) a = func_cref();  // int const &a
    decltype(auto) b = func_ref();   // int &b
    decltype(auto) c = func_val();   // int c
}
