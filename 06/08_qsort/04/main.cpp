#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ticktock.h"
#include <tbb/parallel_invoke.h>

int serial_fib(int n) {
    if (n < 2)
        return n;
    int first = serial_fib(n - 1);
    int second = serial_fib(n - 2);
    return first + second;
}

int fib(int n) {
    if (n < 29)
        return serial_fib(n);
    int first, second;
    tbb::parallel_invoke([&] {
        first = fib(n - 1);
    }, [&] {
        second = fib(n - 2);
    });
    return first + second;
}

int main() {
    TICK(fib);
    std::cout << fib(39) << std::endl;
    TOCK(fib);
    return 0;
}
