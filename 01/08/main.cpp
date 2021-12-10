#include <cstdio>

#include "hello.h"

int main() {
    MyClass mc;
    mc.m_number = 42;
    hello(mc);
    return 0;
}
