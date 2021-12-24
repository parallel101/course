#include <cstdio>

void say_hello() {
    printf("Hello!\n");
}

void call_twice(void func()) {
    func();
    func();
}

int main() {
    call_twice(say_hello);
    return 0;
}
