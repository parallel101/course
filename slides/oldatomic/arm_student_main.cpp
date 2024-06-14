#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"

std::atomic<int> ready = 0;
int answer = 0;
int result;

void t1() {
    answer = 42;
    ready.store(1, std::memory_order_relaxed);
}

void t2() {
    while (ready.load(std::memory_order_relaxed) == 0);
    result = answer;
}

int main() {
    ParallelPool pool{t1, t2};
    pool.join();
    std::cout << result << '\n';
    return 0;
}
