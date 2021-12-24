#include <cstdio>

struct print_t {
    void operator()(float n) {
        printf("Float %f\n", n);
    }

    void operator()(int n) {
        printf("Int %d\n", n);
    }
};
print_t print;

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
