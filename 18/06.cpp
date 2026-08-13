#include "benchmark.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <span>
#include <sys/mman.h>
#include <vector>

#if not defined(__x86_64__) and not defined(__i386__)
#error "06.cpp requires x86 CLFLUSH and TSC instructions"
#endif

namespace {

constexpr std::size_t kPageBytes = 4096;
constexpr std::size_t kHugePageBytes = 2U << 20;
constexpr std::size_t kCandidateWindowBytes = kHugePageBytes;
constexpr std::size_t kCandidateCount =
    kCandidateWindowBytes / kPageBytes - 1;
constexpr std::size_t kDiscoveryTrials = 17;
constexpr std::size_t kFinalistCount = 16;
constexpr std::size_t kValidationTrials = 101;
constexpr std::size_t kPairsPerSample = 101;
constexpr std::size_t kWarmupPairs = 32;
constexpr std::size_t kSamples = 21;

static_assert(kHugePageBytes % kPageBytes == 0);

struct HugePageBuffer {
    explicit HugePageBuffer(std::size_t const bytes)
        : mappingBytes{bytes + kHugePageBytes},
          mapping{mmap(nullptr, mappingBytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)}
    {
        if (mapping == MAP_FAILED) {
            std::perror("mmap");
            std::exit(1);
        }

        auto const address = reinterpret_cast<std::uintptr_t>(mapping);
        auto const alignedAddress =
            (address + kHugePageBytes - 1) & ~(kHugePageBytes - 1);
        values = reinterpret_cast<std::uint64_t *>(alignedAddress);
        valueBytes = bytes;
        if (madvise(values, valueBytes, MADV_HUGEPAGE) != 0) {
            std::perror("madvise(MADV_HUGEPAGE)");
            std::exit(1);
        }
    }

    HugePageBuffer(HugePageBuffer const &) = delete;
    HugePageBuffer &operator=(HugePageBuffer const &) = delete;
    HugePageBuffer(HugePageBuffer &&) = delete;
    HugePageBuffer &operator=(HugePageBuffer &&) = delete;

    ~HugePageBuffer()
    {
        if (mapping != MAP_FAILED and munmap(mapping, mappingBytes) != 0) {
            std::perror("munmap");
        }
    }

    [[nodiscard]] std::uint64_t *data()
    {
        return values;
    }

    [[nodiscard]] std::size_t sizeBytes() const
    {
        return valueBytes;
    }

    [[nodiscard]] bool collapseHugePages(std::size_t const bytes)
    {
#if defined(MADV_COLLAPSE)
        return madvise(values, bytes, MADV_COLLAPSE) == 0;
#else
        static_cast<void>(bytes);
        return false;
#endif
    }

private:
    std::size_t mappingBytes{};
    void *mapping{MAP_FAILED};
    std::uint64_t *values{};
    std::size_t valueBytes{};
};

struct PairScore {
    std::size_t offsetBytes{};
    double cyclesPerPair{};
};

struct PairResult {
    std::size_t offsetBytes{};
    double medianCycles{};
    double p10Cycles{};
    double p90Cycles{};
};

void faultInPages(HugePageBuffer &buffer)
{
    auto const pageCount = buffer.sizeBytes() / kPageBytes;
    auto *const bytes = reinterpret_cast<std::byte *>(buffer.data());
    for (auto page = std::size_t{0}; page < pageCount; ++page) {
        auto *const value = reinterpret_cast<std::uint64_t *>(
            bytes + page * kPageBytes);
        *value = static_cast<std::uint64_t>(page + 1);
    }
}

std::uint64_t timeColdPair(
    std::uint64_t const *const first, std::uint64_t const *const second)
{
    // Cold, independent loads can overlap when DRAM serves them from different
    // banks. Repeated calls alternate the same two rows, exposing the extra
    // precharge/activate work when both addresses map to one bank.
    _mm_clflush(first);
    _mm_clflush(second);
    _mm_mfence();

    _mm_lfence();
    auto const start = __rdtsc();
    auto firstValue = std::uint64_t{};
    auto secondValue = std::uint64_t{};
#if defined(__GNUC__) || defined(__clang__)
    asm volatile(
        "mov (%2), %0\n\t"
        "mov (%3), %1"
        : "=r"(firstValue), "=r"(secondValue)
        : "r"(first), "r"(second)
        : "memory");
#else
    firstValue = *first;
    secondValue = *second;
#endif
    auto auxiliary = unsigned{};
    auto const stop = __rdtscp(&auxiliary);
    _mm_lfence();

    bench::doNotOptimize(firstValue);
    bench::doNotOptimize(secondValue);
    return stop - start;
}

template <std::size_t Trials>
double medianPairCycles(
    std::uint64_t const *const first, std::uint64_t const *const second)
{
    auto samples = std::array<std::uint64_t, Trials>{};
    for (auto trial = std::size_t{0}; trial < Trials; ++trial) {
        samples[trial] = timeColdPair(first, second);
    }
    auto const middle = samples.begin() + Trials / 2;
    std::nth_element(samples.begin(), middle, samples.end());
    return static_cast<double>(*middle);
}

std::vector<std::size_t> makeCandidateOffsets()
{
    auto offsets = std::vector<std::size_t>{};
    offsets.reserve(kCandidateCount);
    for (auto page = std::size_t{1}; page <= kCandidateCount; ++page) {
        offsets.push_back(page * kPageBytes);
    }
    return offsets;
}

std::vector<PairScore> discoverPairs(
    std::uint64_t const *const base,
    std::span<std::size_t const> const offsets)
{
    // Discovery and final measurement are separate so one interrupted timing
    // cannot become the reported "slow" address merely through selection bias.
    auto scores = std::vector<PairScore>{};
    scores.reserve(offsets.size());
    auto const *const bytes = reinterpret_cast<std::byte const *>(base);
    for (auto const offset : offsets) {
        auto const *const candidate =
            reinterpret_cast<std::uint64_t const *>(bytes + offset);
        scores.push_back({
            .offsetBytes = offset,
            .cyclesPerPair =
                medianPairCycles<kDiscoveryTrials>(base, candidate),
        });
    }
    std::ranges::sort(scores, {}, &PairScore::cyclesPerPair);
    return scores;
}

PairScore validateFinalists(
    std::uint64_t const *const base, std::span<PairScore const> const finalists,
    bool const chooseSlowest)
{
    auto validated = std::vector<PairScore>{};
    validated.reserve(finalists.size());
    auto const *const bytes = reinterpret_cast<std::byte const *>(base);
    for (auto const &finalist : finalists) {
        auto const *const candidate = reinterpret_cast<std::uint64_t const *>(
            bytes + finalist.offsetBytes);
        validated.push_back({
            .offsetBytes = finalist.offsetBytes,
            .cyclesPerPair =
                medianPairCycles<kValidationTrials>(base, candidate),
        });
    }
    std::ranges::sort(validated, {}, &PairScore::cyclesPerPair);
    return chooseSlowest ? validated.back() : validated.front();
}

PairResult benchmarkPair(
    std::uint64_t const *const base, PairScore const &pair)
{
    auto const *const bytes = reinterpret_cast<std::byte const *>(base);
    auto const *const candidate = reinterpret_cast<std::uint64_t const *>(
        bytes + pair.offsetBytes);
    for (auto warmup = std::size_t{0}; warmup < kWarmupPairs; ++warmup) {
        bench::doNotOptimize(timeColdPair(base, candidate));
    }

    auto samples = std::vector<double>{};
    samples.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        samples.push_back(
            medianPairCycles<kPairsPerSample>(base, candidate));
    }
    auto const stats = bench::summarizeSamples(samples);
    return {
        .offsetBytes = pair.offsetBytes,
        .medianCycles = stats.medianNs,
        .p10Cycles = stats.p10Ns,
        .p90Cycles = stats.p90Ns,
    };
}

void printPair(char const *const name, PairResult const &result)
{
    auto const mebibytes = static_cast<double>(result.offsetBytes) /
                           static_cast<double>(1U << 20);
    std::cout << name << '\t' << result.offsetBytes / kPageBytes << '\t'
              << mebibytes << '\t' << result.medianCycles << " ["
              << result.p10Cycles << ", " << result.p90Cycles << "]\n";
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto buffer = HugePageBuffer{kCandidateWindowBytes};
    faultInPages(buffer);
    auto const collapsed = buffer.collapseHugePages(kCandidateWindowBytes);
    auto const offsets = makeCandidateOffsets();
    auto const scores = discoverPairs(buffer.data(), offsets);
    auto const fast = validateFinalists(
        buffer.data(), std::span{scores}.first(kFinalistCount), false);
    auto const slow = validateFinalists(
        buffer.data(), std::span{scores}.last(kFinalistCount), true);
    auto const fastResult = benchmarkPair(buffer.data(), fast);
    auto const slowResult = benchmarkPair(buffer.data(), slow);

    std::cout << "DRAM row-buffer candidate scan: " << kCandidateCount
              << " page-aligned candidates in one 2 MiB virtual window\n"
              << "candidate window collapsed to one physical THP: "
              << (collapsed ? "yes" : "no (timing scan still works)") << '\n'
              << "CLFLUSH forces every independently issued load pair past "
                 "LLC into DRAM\n"
              << "pair\tdelta pages\tdelta MiB\tcycles/pair [p10, p90]\n"
              << std::fixed << std::setprecision(2);
    printPair("low-contention", fastResult);
    printPair("high-contention", slowResult);
    std::cout << "slowdown\t\t\t"
              << slowResult.medianCycles / fastResult.medianCycles << "x\n"
              << "All candidates have the same low 12 address bits. The slow "
                 "pair is a timing-selected\n"
              << "row-conflict candidate; undocumented physical channel/bank "
                 "hashing prevents naming its bank.\n";
}
