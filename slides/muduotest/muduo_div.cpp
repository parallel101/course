#include <cstdio>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

const size_t batch = 64 * 1024;
const size_t warmup = 16;
const size_t repeats = 512;

[[gnu::noinline]] void compute0(int const *month, int *__restrict result) {
    for (size_t i = 0; i < batch; ++i) {
        result[i] = (14 - month[i]) / 12;
    }
}

[[gnu::noinline]] void compute1(int const *month, int *__restrict result) {
    for (size_t i = 0; i < batch; ++i) {
        result[i] = (month[i] < 3) ? 1 : 0;
    }
}

[[gnu::noinline]] void compute2(int const *month, int *__restrict result) {
    static const int lut[13] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < batch; ++i) {
        result[i] = lut[month[i]];
    }
}

int main() {
    std::vector<int> result(batch);
    std::vector<int> month(batch);
    std::mt19937 rng;
    std::generate(month.begin(), month.end(), std::bind(std::uniform_int_distribution<int>(1, 12), std::ref(rng)));
    std::array computes{compute0, compute1, compute2};
    for (size_t mode = 0; mode < std::size(computes); ++mode) {
        for (size_t t = 0; t < warmup; ++t) {
            computes[mode](month.data(), result.data());
        }
        auto t0 = std::chrono::steady_clock::now();
        for (size_t t = 0; t < repeats; ++t) {
            computes[mode](month.data(), result.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "compute" << mode << ": " << std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() << "s\n";
    }
}
