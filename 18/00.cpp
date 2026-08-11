#include "benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kMinColumns = 1008;
constexpr std::size_t kMaxColumns = 1036;
constexpr std::size_t kWarmupIterations = 16;
constexpr std::size_t kIterationsPerSample = 256;
constexpr std::size_t kSamples = 31;

struct BenchmarkResult {
    std::size_t columns{};
    double median_ns{};
    double p10_ns{};
    double p90_ns{};
    double bandwidth_gib_per_second{};
};

[[gnu::noinline]] std::int64_t sum_matrix(
    std::span<int const> matrix,
    std::size_t ni,
    std::size_t nj)
{
    std::int64_t sum = 0;
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            sum += matrix[i * nj + j];
        }
    }
    return sum;
}

BenchmarkResult benchmark(std::span<int const> matrix, std::size_t columns)
{
    auto const element_count = kRows * columns;
    auto const input = matrix.first(element_count);
    auto const expected = std::accumulate(
        input.begin(), input.end(), std::int64_t{0});

    for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
        bench::doNotOptimize(sum_matrix(input, kRows, columns));
    }

    std::vector<double> elapsed_ns;
    elapsed_ns.reserve(kSamples);
    for (std::size_t sample = 0; sample < kSamples; ++sample) {
        auto checksum = std::int64_t{0};
        auto const start = bench::Clock::now();
        for (std::size_t iteration = 0; iteration < kIterationsPerSample;
             ++iteration) {
            checksum += sum_matrix(input, kRows, columns);
            bench::doNotOptimize(checksum);
        }
        auto const stop = bench::Clock::now();

        if (checksum != expected * static_cast<std::int64_t>(kIterationsPerSample)) {
            std::cerr << "wrong result for nj=" << columns << '\n';
            std::exit(1);
        }

        auto const total_ns =
            std::chrono::duration<double, std::nano>(stop - start).count();
        elapsed_ns.push_back(total_ns / kIterationsPerSample);
    }

    auto const stats = bench::summarizeSamples(elapsed_ns);
    auto const median_ns = stats.medianNs;
    auto const bytes = static_cast<double>(element_count * sizeof(int));
    auto const seconds = median_ns * 1e-9;
    auto const gib = bytes / static_cast<double>(std::uint64_t{1} << 30);

    return {
        .columns = columns,
        .median_ns = median_ns,
        .p10_ns = stats.p10Ns,
        .p90_ns = stats.p90Ns,
        .bandwidth_gib_per_second = gib / seconds,
    };
}

} // namespace

int main()
{
    bench::setupEnvironment();

    std::vector<int> matrix(kRows * kMaxColumns);
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        matrix[index] = static_cast<int>(index % 17) - 8;
    }

    std::cout << "ni=" << kRows << ", " << kSamples
              << " samples, median [p10, p90]\n"
              << "nj\tlatency (us)\t\tbandwidth (GiB/s)\n";
    for (std::size_t columns = kMinColumns; columns <= kMaxColumns; ++columns) {
        auto const result = benchmark(matrix, columns);
        std::cout << result.columns << '\t'
                  << std::fixed << std::setprecision(3)
                  << result.median_ns / 1e3 << " ["
                  << result.p10_ns / 1e3 << ", "
                  << result.p90_ns / 1e3 << "]\t"
                  << std::setprecision(2)
                  << result.bandwidth_gib_per_second << '\n';
    }
}
