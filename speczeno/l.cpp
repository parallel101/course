#include <cstdio>

struct MyClass {
    MyClass() {
        printf("MyClass initialized\n");
    }

    void someFunc() {
        printf("MyClass::someFunc called\n");
    }

    ~MyClass() {
        printf("MyClass destroyed\n");
    }
};

static MyClass &getMyClassInstance() {
    static MyClass inst;
    return inst;
}

int main() {
    getMyClassInstance().someFunc();
    return 0;
}
