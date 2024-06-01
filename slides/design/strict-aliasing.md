```cpp
#include <memory>

void outer(int *p);

int func0() {
    int a;
    int b = 42;
    outer(&a);
    return b;
}

int func1() { // may-alias
    struct {
        int a;
        int b = 42;
    } s;
    outer(&s.a);
    return s.b;
}

int func2() { // may-alias
    union {
        int a;
        int b = 42;
    } u;
    outer(&u.a);
    return u.b;
}

int func3() { // may-alias
    int a[2];
    a[1] = 42;
    outer(&a[0]);
    return a[1];
}

int func4(int *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func5(int *pa, int *__restrict pb) {
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func6(int *__restrict pa, int *pb) {
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func7(short *pa, int *pb) {
    *pb = 42;
    *pa = 37;
    return *pb;
}

[[gnu::optimize("-fno-strict-aliasing")]] int func8(short *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func9(char *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func10(unsigned char *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func11(signed char *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func12(std::byte *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = std::byte(37);
    return *pb;
}

int func13(char *__restrict pa, int *pb) {
    *pb = 42;
    *pa = 37;
    return *pb;
}

int func14(unsigned int *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

long long func14(long *pa, long long *pb) {
    *pb = 42;
    *pa = 37;
    return *pb;
}

long long func15(long long *pa, long long *pb) { // may-alias
    *pb = 42;
    *pa = 37;
    return *pb;
}

enum enum_int : int {};

int func16(enum_int *pa, int *pb) {
    *pb = 42;
    *pa = enum_int(37);
    return *pb;
}

[[gnu::optimize("-fno-strict-aliasing")]] int func17(enum_int *pa, int *pb) { // may-alias
    *pb = 42;
    *pa = enum_int(37);
    return *pb;
}
```
