#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>

#if defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace bench {

using Clock = std::chrono::steady_clock;

struct SampleStats {
    double medianNs{};
    double p10Ns{};
    double p90Ns{};
};

template <class T>
inline void doNotOptimize(T const &value)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    static volatile T sink;
    sink = value;
#endif
}

inline SampleStats summarizeSamples(std::span<double> const samples)
{
    if (samples.empty()) {
        std::fputs("benchmark requires at least one sample\n", stderr);
        std::exit(1);
    }

    std::ranges::sort(samples);
    auto const count = samples.size();
    return {
        .medianNs = samples[count / 2],
        .p10Ns = samples[count / 10],
        .p90Ns = samples[count * 9 / 10],
    };
}

inline void warnIfFrequencyScalingEnabled(unsigned int const cpu)
{
#if defined(__linux__)
    auto path = std::array<char, 128>{};
    std::snprintf(
        path.data(), path.size(),
        "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_governor", cpu);
    auto *const file = std::fopen(path.data(), "r");
    if (file == nullptr) {
        return;
    }

    auto governor = std::array<char, 64>{};
    auto const hasGovernor = std::fgets(
        governor.data(), static_cast<int>(governor.size()), file) != nullptr;
    std::fclose(file);
    if (hasGovernor and std::strncmp(governor.data(), "performance", 11) != 0) {
        std::fputs(
            "warning: CPU frequency governor is not performance\n", stderr);
    }
#else
    static_cast<void>(cpu);
#endif
}

inline void setupEnvironment()
{
#if defined(__linux__)
    auto cpu = unsigned{};
    if (getcpu(&cpu, nullptr) != 0) {
        std::perror("getcpu");
        std::exit(1);
    }
    if (cpu >= static_cast<unsigned int>(CPU_SETSIZE)) {
        std::fputs("current CPU exceeds cpu_set_t capacity\n", stderr);
        std::exit(1);
    }

    auto cpuSet = cpu_set_t{};
    CPU_ZERO(&cpuSet);
    CPU_SET(static_cast<int>(cpu), &cpuSet);
    if (sched_setaffinity(0, sizeof(cpuSet), &cpuSet) != 0) {
        std::perror("sched_setaffinity");
        std::exit(1);
    }
    warnIfFrequencyScalingEnabled(cpu);
#elif defined(_WIN32)
    auto const cpu = GetCurrentProcessorNumber();
    auto const mask = DWORD_PTR{1} << cpu;
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        std::fputs("SetThreadAffinityMask failed\n", stderr);
        std::exit(1);
    }
#endif
}

} // namespace bench
