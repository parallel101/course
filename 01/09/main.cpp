#include <cstdio>

#include "hello.h"
#include "goodbye.h"

int main() {
    MyClass mc;
    mc.m_number = 42;
    hello(mc);
    goodbye(mc);
    return 0;
}
