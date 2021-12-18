#include <cstdio>
#include <string>
#include <map>

void func(int &) {
    printf("int &\n");
}

void func(int const &) {
    printf("int const &\n");
}

void func(int &&) {
    printf("int &&\n");
}

int main() {
    int a = 0;
    int *p = &a;
    func(a);     // int &
    func(*p);    // int &
    func(p[a]);  // int &
}
