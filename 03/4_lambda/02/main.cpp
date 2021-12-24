#include <cstdio>

void print_number(int n) {
    printf("Number %d\n", n);
}

void call_twice(void func(int)) {
    func(0);
    func(1);
}

int main() {
    call_twice(print_number);
    return 0;
}
