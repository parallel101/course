#include "benchmark.hpp"

#include <algorithm>
#include <array>
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
constexpr std::size_t kFirstColumn = kFloatsPerCacheLine;
constexpr std::size_t kMaxPixelsPerRow = 256;
constexpr std::size_t kKernelRadius = 4;
constexpr std::size_t kRows = 2 * kKernelRadius + 1;
constexpr std::size_t kAssumedL1Ways = 8;
constexpr std::size_t kConflictRowStride = 1024;
constexpr std::size_t kPaddedRowStride =
    kConflictRowStride + kFloatsPerCacheLine;
constexpr std::size_t kPixelsPerSample = 1U << 20;
constexpr std::size_t kWarmupPixels = 1U << 12;
constexpr std::size_t kSamples = 21;

constexpr float kCenterWeight = 70.0F / 256.0F;
constexpr float kNearWeight = 56.0F / 256.0F;
constexpr float kMiddleWeight = 28.0F / 256.0F;
constexpr float kFarWeight = 8.0F / 256.0F;
constexpr float kEdgeWeight = 1.0F / 256.0F;

static_assert(kConflictRowStride * sizeof(float) == kPageBytes);
static_assert(kPaddedRowStride * sizeof(float) ==
              kPageBytes + kCacheLineBytes);
static_assert(kFirstColumn * sizeof(float) % kCacheLineBytes == 0);
static_assert(kFirstColumn + kMaxPixelsPerRow <= kConflictRowStride);
static_assert(kRows + 1 > kAssumedL1Ways);

struct KernelConfig {
    std::span<float const> input;
    std::span<float> output;
    std::size_t rowStride{};
};

struct BenchmarkResult {
    double medianNsPerPixel{};
    double p10NsPerPixel{};
    double p90NsPerPixel{};
};

struct BenchmarkComparison {
    BenchmarkResult conflict;
    BenchmarkResult padded;
};

struct TileResult {
    std::size_t pixels{};
    std::size_t linesPerRow{};
    std::size_t paddedMaxLinesPerSet{};
    BenchmarkComparison comparison;
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
    PageAlignedFloats(PageAlignedFloats &&) = delete;
    PageAlignedFloats &operator=(PageAlignedFloats &&) = delete;

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
    return kRows * rowStride;
}

std::size_t centerIndex(std::size_t const rowStride)
{
    return kKernelRadius * rowStride + kFirstColumn;
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
        for (auto row = std::size_t{0}; row < kRows; ++row) {
            for (auto column = std::size_t{0}; column < rowStride; ++column) {
                auto const logicalValue = (row * 131 + column * 17) % 251;
                inputStorage.data()[row * rowStride + column] =
                    static_cast<float>(logicalValue) / 251.0F;
            }
        }
        std::fill(config.output.begin(), config.output.end(), 0.0F);
    }

    PageAlignedFloats inputStorage;
    PageAlignedFloats outputStorage;
    KernelConfig config;
};

float gaussianAt(
    std::span<float const> const input, std::size_t const center,
    std::size_t const rowStride)
{
    auto const near = input[center - rowStride] + input[center + rowStride];
    auto const middle =
        input[center - 2 * rowStride] + input[center + 2 * rowStride];
    auto const far =
        input[center - 3 * rowStride] + input[center + 3 * rowStride];
    auto const edge =
        input[center - 4 * rowStride] + input[center + 4 * rowStride];
    return kCenterWeight * input[center] + kNearWeight * near +
           kMiddleWeight * middle + kFarWeight * far + kEdgeWeight * edge;
}

template <std::size_t Pixels>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#endif
void verticalGaussian9(
    float const *__restrict input, float *__restrict output,
    std::size_t const rowStride, std::size_t const repetitions)
{
    static_assert(Pixels % kFloatsPerCacheLine == 0);
    static_assert(kFirstColumn + Pixels <= kConflictRowStride);

    auto const centerOffset = centerIndex(rowStride);
    auto const *const center =
        std::assume_aligned<32>(input + centerOffset);
    auto *const outputLine =
        std::assume_aligned<32>(output + centerOffset);
    auto const *const minusOne =
        std::assume_aligned<32>(center - rowStride);
    auto const *const plusOne =
        std::assume_aligned<32>(center + rowStride);
    auto const *const minusTwo =
        std::assume_aligned<32>(center - 2 * rowStride);
    auto const *const plusTwo =
        std::assume_aligned<32>(center + 2 * rowStride);
    auto const *const minusThree =
        std::assume_aligned<32>(center - 3 * rowStride);
    auto const *const plusThree =
        std::assume_aligned<32>(center + 3 * rowStride);
    auto const *const minusFour =
        std::assume_aligned<32>(center - 4 * rowStride);
    auto const *const plusFour =
        std::assume_aligned<32>(center + 4 * rowStride);

    for (auto repetition = std::size_t{0}; repetition < repetitions;
         ++repetition) {
        for (auto x = std::size_t{0}; x < Pixels; ++x) {
            auto const near = minusOne[x] + plusOne[x];
            auto const middle = minusTwo[x] + plusTwo[x];
            auto const far = minusThree[x] + plusThree[x];
            auto const edge = minusFour[x] + plusFour[x];
            outputLine[x] = kCenterWeight * center[x] +
                            kNearWeight * near + kMiddleWeight * middle +
                            kFarWeight * far + kEdgeWeight * edge;
        }
        bench::doNotOptimize(outputLine[0]);
    }
}

bool isClose(float const actual, float const expected)
{
    auto const tolerance = 1e-5F * std::max(1.0F, std::abs(expected));
    return std::abs(actual - expected) <= tolerance;
}

template <std::size_t Pixels>
void validate(Problem &problem)
{
    auto const first = centerIndex(problem.config.rowStride);
    verticalGaussian9<Pixels>(problem.config.input.data(),
                              problem.config.output.data(),
                              problem.config.rowStride, 1);
    for (auto x = std::size_t{0}; x < Pixels; ++x) {
        auto const expected = gaussianAt(
            problem.config.input, first + x, problem.config.rowStride);
        if (not isClose(problem.config.output[first + x], expected)) {
            std::fputs("wrong convolution result\n", stderr);
            std::exit(1);
        }
    }
}

template <std::size_t Pixels>
double measure(Problem &problem)
{
    static_assert(kPixelsPerSample % Pixels == 0);
    constexpr auto repetitions = kPixelsPerSample / Pixels;
    auto const start = bench::Clock::now();
    verticalGaussian9<Pixels>(problem.config.input.data(),
                              problem.config.output.data(),
                              problem.config.rowStride, repetitions);
    auto const stop = bench::Clock::now();
    auto const elapsedNs =
        std::chrono::duration<double, std::nano>(stop - start).count();
    return elapsedNs / static_cast<double>(kPixelsPerSample);
}

BenchmarkResult summarize(std::vector<double> &samples)
{
    auto const stats = bench::summarizeSamples(samples);
    return {
        .medianNsPerPixel = stats.medianNs,
        .p10NsPerPixel = stats.p10Ns,
        .p90NsPerPixel = stats.p90Ns,
    };
}

template <std::size_t Pixels>
TileResult benchmark(Problem &conflict, Problem &padded)
{
    static_assert(kWarmupPixels % Pixels == 0);
    constexpr auto warmupRepetitions = kWarmupPixels / Pixels;
    validate<Pixels>(conflict);
    validate<Pixels>(padded);
    verticalGaussian9<Pixels>(
        conflict.config.input.data(), conflict.config.output.data(),
        conflict.config.rowStride, warmupRepetitions);
    verticalGaussian9<Pixels>(
        padded.config.input.data(), padded.config.output.data(),
        padded.config.rowStride, warmupRepetitions);

    auto conflictSamples = std::vector<double>{};
    auto paddedSamples = std::vector<double>{};
    conflictSamples.reserve(kSamples);
    paddedSamples.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        if ((sample & 1U) == 0) {
            conflictSamples.push_back(measure<Pixels>(conflict));
            paddedSamples.push_back(measure<Pixels>(padded));
        } else {
            paddedSamples.push_back(measure<Pixels>(padded));
            conflictSamples.push_back(measure<Pixels>(conflict));
        }
    }

    constexpr auto linesPerRow = Pixels / kFloatsPerCacheLine;
    return {
        .pixels = Pixels,
        .linesPerRow = linesPerRow,
        .paddedMaxLinesPerSet = std::min(kRows, linesPerRow) + 1,
        .comparison = {
            .conflict = summarize(conflictSamples),
            .padded = summarize(paddedSamples),
        },
    };
}

void printResultRow(TileResult const &result)
{
    auto const &conflictResult = result.comparison.conflict;
    auto const &paddedResult = result.comparison.padded;
    std::cout << result.pixels << '\t' << result.linesPerRow << '\t'
              << result.paddedMaxLinesPerSet << '\t'
              << conflictResult.medianNsPerPixel << " ["
              << conflictResult.p10NsPerPixel << ", "
              << conflictResult.p90NsPerPixel << "]\t"
              << paddedResult.medianNsPerPixel << " ["
              << paddedResult.p10NsPerPixel << ", "
              << paddedResult.p90NsPerPixel << "]\t"
              << conflictResult.medianNsPerPixel /
                     paddedResult.medianNsPerPixel
              << "x\n";
}

void printResults(std::span<TileResult const> const results)
{
    std::cout
        << "9-tap separable Gaussian vertical pass; " << kPixelsPerSample
        << " pixels/layout/sample\n"
        << "tile\tlines\tpadded max lines/set\t4096 B ns/pixel [p10, p90]"
           "\t4160 B ns/pixel [p10, p90]\tslowdown\n"
        << std::fixed << std::setprecision(3);
    for (auto const &result : results) {
        printResultRow(result);
    }
    std::cout << "4096 B always places 9 corresponding source lines plus "
                 "output in one set: 10 lines > "
              << kAssumedL1Ways << " ways.\n"
              << "with 64 B row padding, max occupancy is min(9, tile lines) "
                 "+ output; it exceeds "
              << kAssumedL1Ways << " ways from 128 pixels onward.\n";
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto conflict = Problem{kConflictRowStride};
    auto padded = Problem{kPaddedRowStride};
    auto const results = std::array{
        benchmark<16>(conflict, padded),
        benchmark<32>(conflict, padded),
        benchmark<64>(conflict, padded),
        benchmark<128>(conflict, padded),
        benchmark<256>(conflict, padded),
    };
    printResults(results);
}
