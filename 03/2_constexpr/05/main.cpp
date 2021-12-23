#include <iostream>

template <class T = void>
void func_that_never_pass_compile() {
    "字符串" = 2333;
}

int main() {
    return 0;
}
