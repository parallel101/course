#include <cstdio>

template <class Func>
void call_twice(Func func) {
    func(0);
    func(1);
}

int main() {
    auto myfunc = [] (int n) {
        printf("Number %d\n", n);
    };
    call_twice(myfunc);
    return 0;
}
