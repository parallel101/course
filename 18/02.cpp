#include "benchmark.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kPageBytes = 4096;
constexpr std::size_t kElementsPerPage = kPageBytes / sizeof(int);
constexpr std::size_t kGapPages = 4;
constexpr std::size_t kRepetitions = 4096;
constexpr std::size_t kWarmupSamples = 3;
constexpr std::size_t kSamples = 21;
constexpr auto kLoadOffsets = std::array<std::size_t, 7>{0, 1, 2, 4, 8, 16, 32};

static_assert(kPageBytes % sizeof(int) == 0);
static_assert((kElementsPerPage & (kElementsPerPage - 1)) == 0);

struct KernelConfig {
    std::span<int> stores;
    std::span<int const> loads;
    std::size_t repetitions{};
};

struct BenchmarkResult {
    std::size_t loadOffsetBytes{};
    double medianNsPerPair{};
    double p10NsPerPair{};
    double p90NsPerPair{};
};

#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline, gnu::optimize("no-tree-vectorize", "no-unroll-loops")]]
#endif
// The loaded index selects both addresses in the next iteration. This dependency
// chain prevents out-of-order execution from hiding a replayed aliasing load.
std::uint64_t storeThenLoad(KernelConfig const &config)
{
    auto index = std::size_t{0};
    for (auto repetition = std::size_t{0}; repetition < config.repetitions;
         ++repetition) {
        for (auto step = std::size_t{0}; step < config.stores.size(); ++step) {
            config.stores[index] = static_cast<int>(step + repetition);
            index = static_cast<std::size_t>(config.loads[index]);
        }
    }
    return index;
}

std::size_t pageOffset(void const *const pointer)
{
    return reinterpret_cast<std::uintptr_t>(pointer) % kPageBytes;
}

BenchmarkResult benchmark(KernelConfig const &config)
{
    for (auto sample = std::size_t{0}; sample < kWarmupSamples; ++sample) {
        bench::doNotOptimize(storeThenLoad(config));
    }

    auto elapsedNsPerPair = std::vector<double>{};
    elapsedNsPerPair.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        auto const start = bench::Clock::now();
        auto const checksum = storeThenLoad(config);
        auto const stop = bench::Clock::now();
        if (checksum != 0) {
            std::cerr << "wrong checksum\n";
            std::exit(1);
        }
        bench::doNotOptimize(checksum);

        auto const elapsedNs =
            std::chrono::duration<double, std::nano>(stop - start).count();
        auto const pairCount =
            static_cast<double>(config.stores.size() * config.repetitions);
        elapsedNsPerPair.push_back(elapsedNs / pairCount);
    }

    auto const stats = bench::summarizeSamples(elapsedNsPerPair);
    return {
        .loadOffsetBytes = pageOffset(config.loads.data()),
        .medianNsPerPair = stats.medianNs,
        .p10NsPerPair = stats.p10Ns,
        .p90NsPerPair = stats.p90Ns,
    };
}

int *alignToPage(std::span<int> const storage)
{
    auto const address = reinterpret_cast<std::uintptr_t>(storage.data());
    auto const bytesToAlignment =
        (kPageBytes - address % kPageBytes) % kPageBytes;
    assert(bytesToAlignment % sizeof(int) == 0);
    return storage.data() + bytesToAlignment / sizeof(int);
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto const maxLoadOffset = kLoadOffsets.back();
    auto storage = std::vector<int>(
        kElementsPerPage + kGapPages * kElementsPerPage +
        kElementsPerPage + maxLoadOffset);
    auto *const storePage = alignToPage(storage);
    auto *const loadPage = storePage + kGapPages * kElementsPerPage;
    assert(pageOffset(storePage) == 0);
    assert(pageOffset(loadPage) == 0);

    auto results = std::vector<BenchmarkResult>{};
    results.reserve(kLoadOffsets.size());
    for (auto const loadOffset : kLoadOffsets) {
        for (auto index = std::size_t{0}; index < kElementsPerPage; ++index) {
            loadPage[loadOffset + index] =
                static_cast<int>((index + 1) & (kElementsPerPage - 1));
        }
        auto const config = KernelConfig{
            .stores = {storePage, kElementsPerPage},
            .loads = {loadPage + loadOffset, kElementsPerPage},
            .repetitions = kRepetitions,
        };
        results.push_back(benchmark(config));
    }

    auto const baselineNs = results[1].medianNsPerPair;
    std::cout << "store low12=" << pageOffset(storePage)
              << ", arrays are " << kGapPages * kPageBytes
              << " bytes apart\n"
              << "load low12\tdelta\tns/store+load [p10, p90]\tslowdown\n";
    for (auto resultIndex = std::size_t{0}; resultIndex < results.size();
         ++resultIndex) {
        auto const &result = results[resultIndex];

        std::cout << result.loadOffsetBytes << '\t'
                  << kLoadOffsets[resultIndex] * sizeof(int) << '\t'
                  << std::fixed << std::setprecision(3)
                  << result.medianNsPerPair << " ["
                  << result.p10NsPerPair << ", "
                  << result.p90NsPerPair << "]\t"
                  << result.medianNsPerPair / baselineNs << "x\n";
    }
}
