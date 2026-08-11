#include "benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 1024;
constexpr std::size_t kMaxPadding = 16;
constexpr std::size_t kWarmupIterations = 16;
constexpr std::size_t kIterationsPerSample = 256;
constexpr std::size_t kSamples = 31;

struct MatrixView {
    std::span<int const> values;
    std::size_t rows{};
    std::size_t columns{};
    std::size_t rowStride{};
};

struct BenchmarkResult {
    std::size_t rowStride{};
    double medianNs{};
    double p10Ns{};
    double p90Ns{};
    double usefulBandwidthGiBPerSecond{};
};

[[gnu::noinline]] std::int64_t sumColumns(MatrixView const &matrix)
{
    auto sum = std::int64_t{0};
    for (auto j = std::size_t{0}; j < matrix.columns; ++j) {
        for (auto i = std::size_t{0}; i < matrix.rows; ++i) {
            sum += matrix.values[i * matrix.rowStride + j];
        }
    }
    return sum;
}

std::int64_t referenceSum(MatrixView const &matrix)
{
    auto sum = std::int64_t{0};
    for (auto i = std::size_t{0}; i < matrix.rows; ++i) {
        for (auto j = std::size_t{0}; j < matrix.columns; ++j) {
            sum += matrix.values[i * matrix.rowStride + j];
        }
    }
    return sum;
}

std::vector<int> makeMatrix(std::size_t const rowStride)
{
    auto matrix = std::vector<int>(kRows * rowStride, 0x5a5a5a5a);
    for (auto i = std::size_t{0}; i < kRows; ++i) {
        for (auto j = std::size_t{0}; j < kColumns; ++j) {
            auto const logicalIndex = i * kColumns + j;
            matrix[i * rowStride + j] =
                static_cast<int>(logicalIndex % 17) - 8;
        }
    }
    return matrix;
}

BenchmarkResult benchmark(MatrixView const &matrix)
{
    auto const expected = referenceSum(matrix);
    for (auto iteration = std::size_t{0}; iteration < kWarmupIterations;
         ++iteration) {
        bench::doNotOptimize(sumColumns(matrix));
    }

    auto elapsedNs = std::vector<double>{};
    elapsedNs.reserve(kSamples);
    for (auto sample = std::size_t{0}; sample < kSamples; ++sample) {
        auto checksum = std::int64_t{0};
        auto const start = bench::Clock::now();
        for (auto iteration = std::size_t{0}; iteration < kIterationsPerSample;
             ++iteration) {
            checksum += sumColumns(matrix);
            bench::doNotOptimize(checksum);
        }
        auto const stop = bench::Clock::now();

        auto const expectedChecksum =
            expected * static_cast<std::int64_t>(kIterationsPerSample);
        if (checksum != expectedChecksum) {
            std::cerr << "wrong result for row stride " << matrix.rowStride
                      << '\n';
            std::exit(1);
        }

        auto const totalNs =
            std::chrono::duration<double, std::nano>(stop - start).count();
        elapsedNs.push_back(totalNs / kIterationsPerSample);
    }

    auto const stats = bench::summarizeSamples(elapsedNs);
    auto const medianNs = stats.medianNs;
    auto const usefulBytes =
        static_cast<double>(matrix.rows * matrix.columns * sizeof(int));
    auto const seconds = medianNs * 1e-9;
    auto const gib = usefulBytes /
                     static_cast<double>(std::uint64_t{1} << 30);

    return {
        .rowStride = matrix.rowStride,
        .medianNs = medianNs,
        .p10Ns = stats.p10Ns,
        .p90Ns = stats.p90Ns,
        .usefulBandwidthGiBPerSecond = gib / seconds,
    };
}

void printResult(BenchmarkResult const &result, double const baselineNs)
{
    auto const padding = result.rowStride - kColumns;
    std::cout << result.rowStride << '\t'
              << result.rowStride * sizeof(int) << '\t'
              << padding << '\t'
              << std::fixed << std::setprecision(3)
              << result.medianNs / 1e3 << " ["
              << result.p10Ns / 1e3 << ", "
              << result.p90Ns / 1e3 << "]\t"
              << std::setprecision(2)
              << result.usefulBandwidthGiBPerSecond << '\t'
              << baselineNs / result.medianNs << "x\n";
}

} // namespace

int main()
{
    bench::setupEnvironment();

    auto results = std::vector<BenchmarkResult>{};
    results.reserve(kMaxPadding + 1);
    for (auto padding = std::size_t{0}; padding <= kMaxPadding; ++padding) {
        auto const rowStride = kColumns + padding;
        auto const storage = makeMatrix(rowStride);
        auto const matrix = MatrixView{
            .values = storage,
            .rows = kRows,
            .columns = kColumns,
            .rowStride = rowStride,
        };
        results.push_back(benchmark(matrix));
    }

    std::cout << "logical matrix=" << kRows << 'x' << kColumns << ", "
              << kSamples << " samples, median [p10, p90]\n"
              << "stride\tbytes\tpadding\tlatency (us)\t\t"
                 "useful GiB/s\tspeedup\n";
    auto const baselineNs = results.front().medianNs;
    for (auto const &result : results) {
        printResult(result, baselineNs);
    }
}
