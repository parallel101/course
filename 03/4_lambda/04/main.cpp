#include <cstdio>

void print(float n) {
    printf("Float %f\n", n);
}

void print(int n) {
    printf("Int %d\n", n);
}

template <class Func>
void call_twice(Func func) {
    func(0);
    func(1);
    func(3.14f);
}

int main() {
    call_twice(print);
    return 0;
}
