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
constexpr std::size_t kFirstColumn = kFloatsPerCacheLine;
constexpr std::size_t kPixelsPerRow = 32;
constexpr std::size_t kLinesPerTile =
    kPixelsPerRow / kFloatsPerCacheLine;
constexpr std::size_t kKernelRadius = 4;
constexpr std::size_t kRows = 2 * kKernelRadius + 1;
constexpr std::size_t kAssumedL1Ways = 8;
constexpr std::size_t kConflictRowStride = 1024;
constexpr std::size_t kPaddedRowStride =
    kConflictRowStride + kFloatsPerCacheLine;
constexpr std::size_t kIterationsPerSample = 32768;
constexpr std::size_t kWarmupIterations = 16;
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
static_assert(kPixelsPerRow % kFloatsPerCacheLine == 0);
static_assert(kFirstColumn + kPixelsPerRow <= kConflictRowStride);
static_assert(kRows + 1 > kAssumedL1Ways);
static_assert(kLinesPerTile + 1 <= kAssumedL1Ways);

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

#if defined(__GNUC__) || defined(__clang__)
[[gnu::noinline]]
#endif
void verticalGaussian9(
    float const *__restrict input, float *__restrict output,
    std::size_t const rowStride, std::size_t const repetitions)
{
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
        for (auto x = std::size_t{0}; x < kPixelsPerRow; ++x) {
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

void validate(Problem &problem)
{
    auto const first = centerIndex(problem.config.rowStride);
    verticalGaussian9(problem.config.input.data(), problem.config.output.data(),
                      problem.config.rowStride, 1);
    for (auto x = std::size_t{0}; x < kPixelsPerRow; ++x) {
        auto const expected = gaussianAt(
            problem.config.input, first + x, problem.config.rowStride);
        if (not isClose(problem.config.output[first + x], expected)) {
            std::fputs("wrong convolution result\n", stderr);
            std::exit(1);
        }
    }
}

double measure(Problem &problem)
{
    auto const start = bench::Clock::now();
    verticalGaussian9(problem.config.input.data(), problem.config.output.data(),
                      problem.config.rowStride, kIterationsPerSample);
    auto const stop = bench::Clock::now();
    auto const elapsedNs =
        std::chrono::duration<double, std::nano>(stop - start).count();
    auto const pixels =
        static_cast<double>(kPixelsPerRow * kIterationsPerSample);
    return elapsedNs / pixels;
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

BenchmarkComparison benchmark(Problem &conflict, Problem &padded)
{
    validate(conflict);
    validate(padded);
    verticalGaussian9(conflict.config.input.data(),
                      conflict.config.output.data(), conflict.config.rowStride,
                      kWarmupIterations);
    verticalGaussian9(padded.config.input.data(), padded.config.output.data(),
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

    std::cout << "9-tap separable Gaussian vertical pass, " << kPixelsPerRow
              << " aligned float pixels\n"
              << "row stride\trow bytes\tns/pixel [p10, p90]\tslowdown\n"
              << kConflictRowStride << '\t'
              << kConflictRowStride * sizeof(float) << '\t' << std::fixed
              << std::setprecision(3) << conflictResult.medianNsPerPixel
              << " [" << conflictResult.p10NsPerPixel << ", "
              << conflictResult.p90NsPerPixel << "]\t"
              << conflictResult.medianNsPerPixel /
                     paddedResult.medianNsPerPixel
              << "x\n"
              << kPaddedRowStride << '\t'
              << kPaddedRowStride * sizeof(float) << '\t'
              << paddedResult.medianNsPerPixel << " ["
              << paddedResult.p10NsPerPixel << ", "
              << paddedResult.p90NsPerPixel << "]\t1.000x\n"
              << "4096 B pitch maps each group of 9 corresponding source "
                 "lines plus output to one set: 10 lines > "
              << kAssumedL1Ways << " ways.\n"
              << "the tile spans " << kLinesPerTile
              << " lines/row; 64 B padding rotates rows and caps the padded "
                 "case at "
              << kLinesPerTile + 1 << " lines/set.\n";
}
