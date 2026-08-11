#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kPageBytes = 4096;
constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kFloatsPerCacheLine = kCacheLineBytes / sizeof(float);
constexpr std::size_t kFirstColumn = 8;
constexpr std::size_t kUpdatesPerRow = 256;
constexpr std::size_t kRows = 5;
constexpr std::size_t kPlanes = 5;
constexpr std::size_t kStencilRadius = 2;
constexpr std::size_t kConflictRowStride = 1024;
constexpr std::size_t kPaddedRowStride =
    kConflictRowStride + kFloatsPerCacheLine;
constexpr std::size_t kIterationsPerSample = 2048;
constexpr std::size_t kWarmupIterations = 8;
constexpr std::size_t kSamples = 21;

static_assert(kConflictRowStride * sizeof(float) == kPageBytes);
static_assert(kPaddedRowStride * sizeof(float) == kPageBytes + kCacheLineBytes);
static_assert(kFirstColumn * sizeof(float) % 32 == 0);
static_assert(kFirstColumn + kUpdatesPerRow + kStencilRadius <=
              kConflictRowStride);
static_assert(kPlanes > 2 * kStencilRadius);
static_assert(kRows > 2 * kStencilRadius);

struct KernelConfig {
    std::span<float const> input;
    std::span<float> output;
    std::size_t rowStride{};
};

struct BenchmarkResult {
    double medianNsPerUpdate{};
    double p10NsPerUpdate{};
    double p90NsPerUpdate{};
};

struct BenchmarkComparison {
    BenchmarkResult conflict;
    BenchmarkResult padded;
};

struct PageAlignedFloats {
    explicit PageAlignedFloats(std::size_t const elements)
        : values{static_cast<float *>(std::aligned_alloc(
              kPageBytes, ((elements * sizeof(float) + kPageBytes - 1) /
                           kPageBytes) *
                              kPageBytes))}
    {
        if (values == nullptr) {
            std::perror("aligned_alloc");
            std::exit(1);
        }
    }

    PageAlignedFloats(PageAlignedFloats const &) = delete;
    PageAlignedFloats &operator=(PageAlignedFloats const &) = delete;

    ~PageAlignedFloats()
    {
        std::free(values);
    }

    [[nodiscard]] float *data()
    {
        return values;
    }

private:
    float *values{};
};

std::size_t elementCount(std::size_t const rowStride)
{
    return kPlanes * kRows * rowStride;
}

std::size_t gridIndex(
    std::size_t const rowStride, std::size_t const z, std::size_t const y,
    std::size_t const x)
{
    return (z * kRows + y) * rowStride + x;
}

struct Problem {
    explicit Problem(std::size_t const rowStride)
        : inputStorage{elementCount(rowStride)},
          outputStorage{elementCount(rowStride)},
          config{
              .input = {inputStorage.data(), elementCount(rowStride)},
              .output = {outputStorage.data(), elementCount(rowStride)},
              .rowStride = rowStride,
          }
    {
        for (auto index = std::size_t{0}; index < config.input.size(); ++index) {
            inputStorage.data()[index] =
                static_cast<float>((index * 17) % 101) * 0.01F;
        }
        std::fill(config.output.begin(), config.output.end(), 0.0F);
    }

    PageAlignedFloats inputStorage;
    PageAlignedFloats outputStorage;
    KernelConfig config;
};

float laplacianAt(
    std::span<float const> const input, std::size_t const center,
    std::size_t const rowStride)
{
    auto const planeStride = kRows * rowStride;
    return -90.0F * input[center] +
           16.0F * (input[center - 1] + input[center + 1] +
                     input[center - rowStride] + input[center + rowStride] +
                     input[center - planeStride] + input[center + planeStride]) -
           (input[center - 2] + input[center + 2] +
            input[center - 2 * rowStride] + input[center + 2 * rowStride] +
            input[center - 2 * planeStride] + input[center + 2 * planeStride]);
}

#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#endif
void laplacian13(
    float const *__restrict input, float *__restrict output,
    std::size_t const rowStride, std::size_t const repetitions)
{
    auto const planeStride = kRows * rowStride;
    auto const centerIndex =
        gridIndex(rowStride, kStencilRadius, kStencilRadius, kFirstColumn);
    auto const *const center =
        std::assume_aligned<32>(input + centerIndex);
    auto *const outputLine =
        std::assume_aligned<32>(output + centerIndex);
    auto const *const xMinusOne = center - 1;
    auto const *const xPlusOne = center + 1;
    auto const *const xMinusTwo = center - 2;
    auto const *const xPlusTwo = center + 2;
    auto const *const yMinusOne =
        std::assume_aligned<32>(center - rowStride);
    auto const *const yPlusOne =
        std::assume_aligned<32>(center + rowStride);
    auto const *const yMinusTwo =
        std::assume_aligned<32>(center - 2 * rowStride);
    auto const *const yPlusTwo =
        std::assume_aligned<32>(center + 2 * rowStride);
    auto const *const zMinusOne =
        std::assume_aligned<32>(center - planeStride);
    auto const *const zPlusOne =
        std::assume_aligned<32>(center + planeStride);
    auto const *const zMinusTwo =
        std::assume_aligned<32>(center - 2 * planeStride);
    auto const *const zPlusTwo =
        std::assume_aligned<32>(center + 2 * planeStride);

    for (auto repetition = std::size_t{0}; repetition < repetitions;
         ++repetition) {
        for (auto x = std::size_t{0}; x < kUpdatesPerRow; ++x) {
            auto const near = xMinusOne[x] + xPlusOne[x] + yMinusOne[x] +
                              yPlusOne[x] + zMinusOne[x] + zPlusOne[x];
            auto const far = xMinusTwo[x] + xPlusTwo[x] + yMinusTwo[x] +
                             yPlusTwo[x] + zMinusTwo[x] + zPlusTwo[x];
            outputLine[x] = -90.0F * center[x] + 16.0F * near - far;
        }
        bench::doNotOptimize(outputLine[0]);
    }
}

bool isClose(float const actual, float const expected)
{
    auto const tolerance = 1e-5F * std::max(1.0F, std::abs(expected));
    return std::abs(actual - expected) <= tolerance;
}

void validate(Problem &problem)
{
    auto const referenceIndex =
        gridIndex(problem.config.rowStride, kStencilRadius, kStencilRadius,
                  kFirstColumn);
    auto const expected = laplacianAt(
        problem.config.input, referenceIndex, problem.config.rowStride);
    laplacian13(problem.config.input.data(), problem.config.output.data(),
                problem.config.rowStride, 1);
    if (not isClose(problem.config.output[referenceIndex], expected)) {
        std::fputs("wrong stencil result\n", stderr);
        std::exit(1);
    }
}

double measure(Problem &problem)
{
    auto const start = bench::Clock::now();
    laplacian13(problem.config.input.data(), problem.config.output.data(),
                problem.config.rowStride, kIterationsPerSample);
    auto const stop = bench::Clock::now();
    auto const elapsedNs =
        std::chrono::duration<double, std::nano>(stop - start).count();
    auto const updates =
        static_cast<double>(kUpdatesPerRow * kIterationsPerSample);
    return elapsedNs / updates;
}

BenchmarkResult summarize(std::vector<double> &samples)
{
    auto const stats = bench::summarizeSamples(samples);
    return {
        .medianNsPerUpdate = stats.medianNs,
        .p10NsPerUpdate = stats.p10Ns,
        .p90NsPerUpdate = stats.p90Ns,
    };
}

BenchmarkComparison benchmark(Problem &conflict, Problem &padded)
{
    validate(conflict);
    validate(padded);
    laplacian13(conflict.config.input.data(), conflict.config.output.data(),
                conflict.config.rowStride, kWarmupIterations);
    laplacian13(padded.config.input.data(), padded.config.output.data(),
                padded.config.rowStride, kWarmupIterations);

    auto conflictSamples = std::vector<double>{};
    auto paddedSamples = std::vector<double>{};
    conflictSamples.reserve(kSamples);
    paddedSamples.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        if ((sample & 1U) == 0) {
            conflictSamples.push_back(measure(conflict));
            paddedSamples.push_back(measure(padded));
        } else {
            paddedSamples.push_back(measure(padded));
            conflictSamples.push_back(measure(conflict));
        }
    }

    return {
        .conflict = summarize(conflictSamples),
        .padded = summarize(paddedSamples),
    };
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto conflict = Problem{kConflictRowStride};
    auto padded = Problem{kPaddedRowStride};
    auto const results = benchmark(conflict, padded);
    auto const &conflictResult = results.conflict;
    auto const &paddedResult = results.padded;

    std::cout << "fourth-order 13-point 3D Laplacian, " << kUpdatesPerRow
              << " aligned updates on a " << kRows << 'x' << kPlanes
              << " yz slab\n"
              << "row stride\trow bytes\tns/update [p10, p90]\tslowdown\n"
              << kConflictRowStride << '\t'
              << kConflictRowStride * sizeof(float) << '\t' << std::fixed
              << std::setprecision(3) << conflictResult.medianNsPerUpdate
              << " [" << conflictResult.p10NsPerUpdate << ", "
              << conflictResult.p90NsPerUpdate << "]\t"
              << conflictResult.medianNsPerUpdate /
                     paddedResult.medianNsPerUpdate
              << "x\n"
              << kPaddedRowStride << '\t'
              << kPaddedRowStride * sizeof(float) << '\t'
              << paddedResult.medianNsPerUpdate << " ["
              << paddedResult.p10NsPerUpdate << ", "
              << paddedResult.p90NsPerUpdate << "]\t1.000x\n"
              << "1024 floats = 4096 B = 64 cache lines: all 9 input-line "
                 "groups share an L1 set; the page-aligned output adds a 10th "
                 "line.\n"
              << "padding one 64 B line makes neighbouring rows differ by one "
                 "set; planes differ by "
              << kRows << " sets.\n";
}
