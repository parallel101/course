#include <list>
#include <vector>
#include <array>
#include <memory_resource>
#include <iostream>

template <typename Func>
auto benchmark(Func test_func, int iterations) {
    const auto start = std::chrono::system_clock::now();
    while (iterations-- > 0) { test_func(); }
    const auto stop = std::chrono::system_clock::now();
    const auto secs = std::chrono::duration<double>(stop - start);
    return secs.count();
}

int main() {
    constexpr int kNodes = 250000;
    constexpr int kRows = 500;
    constexpr int kCols = 500;
    constexpr int kIters = 100;

    std::cout << "list=" << benchmark([&] {
        std::list<int> a;
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "list_pmr=" << benchmark([&] {
        std::list<int, std::pmr::polymorphic_allocator<int>> a;
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "list_pmr_nobuf=" << benchmark([&] {
        std::pmr::monotonic_buffer_resource mbr;
        std::list<int, std::pmr::polymorphic_allocator<int>> a(&mbr);
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "list_pmr_buf=" << benchmark([&] {
        std::array<std::byte, kNodes * 32> buf;
        std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
        std::list<int, std::pmr::polymorphic_allocator<int>> a(&mbr);
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "vector=" << benchmark([&] {
        std::vector<int> a;
        a.reserve(kNodes);
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "vector_pmr=" << benchmark([&] {
        std::vector<int, std::pmr::polymorphic_allocator<int>> a;
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "vector_pmr_nobuf=" << benchmark([&] {
        std::pmr::monotonic_buffer_resource mbr;
        std::pmr::vector<int> a(&mbr);
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "vector_pmr_buf=" << benchmark([&] {
        std::array<std::byte, kNodes * 32> buf;
        std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
        std::pmr::vector<int> a(&mbr);
        for (int i = 0; i < kNodes; i++) {
            a.push_back(i);
        }
    }, kIters) << '\n';

    std::cout << "vector_vector=" << benchmark([&] {
        std::vector<std::vector<int>> a;
        for (int i = 0; i < kCols; i++) {
            auto &b = a.emplace_back();
            for (int j = 0; j < kRows; j++) {
                b.push_back(j);
            }
        }
    }, kIters) << '\n';

    std::cout << "vector_vector_pmr=" << benchmark([&] {
        std::pmr::vector<std::pmr::vector<int>> a;
        for (int i = 0; i < kCols; i++) {
            auto &b = a.emplace_back();
            for (int j = 0; j < kRows; j++) {
                b.push_back(j);
            }
        }
    }, kIters) << '\n';

    std::cout << "vector_vector_pmr_nobuf=" << benchmark([&] {
        std::pmr::monotonic_buffer_resource mbr;
        std::pmr::vector<std::pmr::vector<int>> a(&mbr);
        for (int i = 0; i < kCols; i++) {
            auto &b = a.emplace_back();
            for (int j = 0; j < kRows; j++) {
                b.push_back(j);
            }
        }
    }, kIters) << '\n';

    std::cout << "vector_vector_pmr_buf=" << benchmark([&] {
        std::array<std::byte, kRows * kCols * 32> buf;
        std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
        std::pmr::vector<std::pmr::vector<int>> a(&mbr);
        for (int i = 0; i < kCols; i++) {
            auto &b = a.emplace_back();
            for (int j = 0; j < kRows; j++) {
                b.push_back(j);
            }
        }
    }, kIters) << '\n';

    return 0;
}
