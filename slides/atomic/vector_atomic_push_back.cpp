#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"

std::vector<int> a(10);
std::atomic<int> counter;

void t1() {
    a[counter.fetch_add(1, std::memory_order_relaxed)] = 1;
    a[counter.fetch_add(1, std::memory_order_relaxed)] = 3;
    a[counter.fetch_add(1, std::memory_order_relaxed)] = 4;
}

void t2() {
    a[counter.fetch_add(1, std::memory_order_relaxed)] = 2;
}

int main() {
    ParallelPool pool{t1, t2};
    pool.join();
    a.resize(counter.load(std::memory_order_relaxed));
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::cout << a[i] << ' ';
    }
    return 0;
}
