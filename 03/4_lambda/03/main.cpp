#include <cstdio>

void print_float(float n) {
    printf("Float %f\n", n);
}

void print_int(int n) {
    printf("Int %d\n", n);
}

template <class Func>
void call_twice(Func func) {
    func(0);
    func(1);
}

int main() {
    call_twice(print_float);
    call_twice(print_int);
    return 0;
}
