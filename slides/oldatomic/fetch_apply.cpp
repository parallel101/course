#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"

std::atomic<int> counter{2};

template <class T, class Func>
T fetch_apply(std::atomic<T> &dest, Func &&func, std::memory_order m = std::memory_order_seq_cst) {
    T oldc, newc;
    oldc = dest.load();
    do {
        newc = func(oldc);
    } while (!dest.compare_exchange_weak(oldc, newc, m, std::memory_order_relaxed));
    return oldc;
}

void t1() {
    fetch_apply(counter, [] (int c) { return c * 2; }, std::memory_order_relaxed);
}

void t2() {
    fetch_apply(counter, [] (int c) { return c * 4; }, std::memory_order_relaxed);
}

int main() {
    ParallelPool pool{t1, t2};
    pool.join();
    std::cout << counter << '\n';
    return 0;
}
