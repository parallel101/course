#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ticktock.h"
#include <tbb/task_group.h>

int fib(int n) {
    if (n < 2)
        return n;
    int first, second;
    tbb::task_group tg;
    tg.run([&] {
        first = fib(n - 1);
    });
    tg.run([&] {
        second = fib(n - 2);
    });
    tg.wait();
    return first + second;
}

int main() {
    TICK(fib);
    std::cout << fib(39) << std::endl;
    TOCK(fib);
    return 0;
}
