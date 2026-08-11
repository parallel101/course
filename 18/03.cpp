#include "benchmark.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kPageBytes = 4096;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kL1Associativity = 8;
constexpr std::size_t kStreamCount = kL1Associativity + 1;
constexpr std::size_t kWalkSteps = 8192;
constexpr std::size_t kIterationsPerSample = 32;
constexpr std::size_t kWarmupIterations = 8;
constexpr std::size_t kSamples = 21;

static_assert(kPageBytes % kCacheLineBytes == 0);
static_assert((kStreamCount - 1) * kCacheLineBytes + sizeof(int) <=
              kPageBytes);

struct Page {
    Page()
        : values{static_cast<int *>(std::aligned_alloc(kPageBytes, kPageBytes))}
    {
        if (values == nullptr) {
            std::perror("aligned_alloc");
            std::exit(1);
        }
    }

    Page(Page const &) = delete;
    Page &operator=(Page const &) = delete;

    ~Page()
    {
        std::free(values);
    }

    [[nodiscard]] int *data()
    {
        return values;
    }

private:
    int *values{};
};

struct Result {
    double medianNsPerLoad{};
    double p10NsPerLoad{};
    double p90NsPerLoad{};
};

using Pages = std::array<Page, kStreamCount>;
using Pointers = std::array<int const *, kStreamCount>;

Pointers makePointers(Pages &pages, bool const staggerCacheSets)
{
    auto pointers = Pointers{};
    for (auto stream = std::size_t{0}; stream < kStreamCount; ++stream) {
        auto const offset = staggerCacheSets ? stream * (kCacheLineBytes / sizeof(int)) : 0;
        auto *const pointer = pages[stream].data() + offset;
        *pointer = 0;
        pointers[stream] = pointer;
    }
    return pointers;
}

#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#endif
std::size_t walk(Pointers const &pointers)
{
    auto index = std::size_t{0};
    for (auto step = std::size_t{0}; step < kWalkSteps; ++step) {
        index = static_cast<std::size_t>(pointers[0][index]);
        index = static_cast<std::size_t>(pointers[1][index]);
        index = static_cast<std::size_t>(pointers[2][index]);
        index = static_cast<std::size_t>(pointers[3][index]);
        index = static_cast<std::size_t>(pointers[4][index]);
        index = static_cast<std::size_t>(pointers[5][index]);
        index = static_cast<std::size_t>(pointers[6][index]);
        index = static_cast<std::size_t>(pointers[7][index]);
        index = static_cast<std::size_t>(pointers[8][index]);
    }
    return index;
}

Result benchmark(Pointers const &pointers)
{
    for (auto iteration = std::size_t{0}; iteration < kWarmupIterations;
         ++iteration) {
        bench::doNotOptimize(walk(pointers));
    }

    auto samples = std::vector<double>{};
    samples.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        auto checksum = std::size_t{0};
        auto const start = bench::Clock::now();
        for (auto iteration = std::size_t{0}; iteration < kIterationsPerSample;
             ++iteration) {
            checksum += walk(pointers);
            bench::doNotOptimize(checksum);
        }
        auto const stop = bench::Clock::now();
        if (checksum != 0) {
            std::fputs("wrong checksum\n", stderr);
            std::exit(1);
        }

        auto const elapsedNs =
            std::chrono::duration<double, std::nano>(stop - start).count();
        samples.push_back(elapsedNs /
                          static_cast<double>(kIterationsPerSample * kWalkSteps *
                                              kStreamCount));
    }

    auto const stats = bench::summarizeSamples(samples);
    return {
        .medianNsPerLoad = stats.medianNs,
        .p10NsPerLoad = stats.p10Ns,
        .p90NsPerLoad = stats.p90Ns,
    };
}

std::size_t pageOffset(void const *const pointer)
{
    return reinterpret_cast<std::uintptr_t>(pointer) % kPageBytes;
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto pages = Pages{};
    auto const sameSet = makePointers(pages, false);
    auto const staggered = makePointers(pages, true);
    auto const sameSetResult = benchmark(sameSet);
    auto const staggeredResult = benchmark(staggered);

    std::cout << kStreamCount << " loads compete for an " << kL1Associativity
              << "-way L1D set. All pages are 4 KiB aligned.\n"
              << "same-set page offsets: " << pageOffset(sameSet[0]) << ", "
              << pageOffset(sameSet[8]) << "\n"
              << "staggered page offsets: " << pageOffset(staggered[0]) << ", "
              << pageOffset(staggered[8]) << "\n"
              << "layout\tns/load [p10, p90]\n"
              << "same set\t" << std::fixed << std::setprecision(3)
              << sameSetResult.medianNsPerLoad << " ["
              << sameSetResult.p10NsPerLoad << ", "
              << sameSetResult.p90NsPerLoad << "]\n"
              << "staggered\t" << staggeredResult.medianNsPerLoad << " ["
              << staggeredResult.p10NsPerLoad << ", "
              << staggeredResult.p90NsPerLoad << "]\n"
              << "slowdown\t"
              << sameSetResult.medianNsPerLoad /
                     staggeredResult.medianNsPerLoad
              << "x\n";
}
