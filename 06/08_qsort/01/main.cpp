#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ticktock.h"

int fib(int n) {
    if (n < 2)
        return n;
    int first = fib(n - 1);
    int second = fib(n - 2);
    return first + second;
}

int main() {
    TICK(fib);
    std::cout << fib(39) << std::endl;
    TOCK(fib);
    return 0;
}
