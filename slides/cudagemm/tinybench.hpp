#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#if __x86_64__ || __amd64__
#include <x86intrin.h>
#elif _M_AMD64 || _M_IX86
#include <emmintrin.h>
extern "C" int64_t __rdtsc();
#pragma intrinsic(__rdtsc)
#elif _M_ARM64 || _M_ARM64EC
#include <intrin.h>
#elif __powerpc64__
#include <builtins.h>
#else
#include <chrono>
#endif

namespace tinybench {

#if __GNUC__ && !__clang__
#define TINYBENCH_OPTIMIZE __attribute__((__optimize__("O3")))
#define TINYBENCH_NO_OPTIMIZE __attribute__((__optimize__("O0")))
#elif _MSC_VER && !__clang__
#define TINYBENCH_OPTIMIZE __pragma(optimize("g", on))
#define TINYBENCH_NO_OPTIMIZE __pragma(optimize("g", off))
#else
#define TINYBENCH_OPTIMIZE
#define TINYBENCH_NO_OPTIMIZE
#endif
#if __GNUC__ || __clang__
#define TINYBENCH_ALWAYS_INLINE __attribute__((__always_inline__))
#define TINYBENCH_NOINLINE __attribute__((__noinline__))
#define TINYBENCH_RESTRICT __restrict
#elif _MSC_VER && !__clang__
#define TINYBENCH_ALWAYS_INLINE __forceinline
#define TINYBENCH_NOINLINE __declspec(noinline)
#define TINYBENCH_RESTRICT __restrict
#else
#define TINYBENCH_ALWAYS_INLINE
#define TINYBENCH_NOINLINE
#define TINYBENCH_RESTRICT
#endif

TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE inline void mfence() {
#if __x86_64__ || __amd64__ || _M_AMD64 || _M_IX86
    _mm_mfence();
#elif _M_ARM64 || _M_ARM64EC
    _ReadWriteBarrier();
#elif __aarch64__
    asm volatile("dmb ish" ::: "memory");
#elif __powerpc64__
    __builtin_ppc_isync();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE inline void sfence() {
#if __x86_64__ || __amd64__ || _M_AMD64 || _M_IX86
    _mm_sfence();
#elif _M_ARM64 || _M_ARM64EC
    _WriteBarrier();
#elif __aarch64__
    asm volatile ("dmb ishst" ::: "memory");
#elif __powerpc64__
    __builtin_ppc_isync();
#else
    std::atomic_signal_fence(std::memory_order_release);
#endif
}

TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE inline void lfence() {
#if __x86_64__ || __amd64__ || _M_AMD64 || _M_IX86
    _mm_lfence();
#elif _M_ARM64 || _M_ARM64EC
    _ReadBarrier();
#elif __aarch64__
    asm volatile ("isb" ::: "memory");
#elif __powerpc64__
    __builtin_ppc_isync();
#else
    std::atomic_signal_fence(std::memory_order_acquire);
#endif
}

TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE inline int64_t now() {
#if __x86_64__ || __amd64__ || _M_AMD64 || _M_IX86
    int64_t t = __rdtsc();
    return t;
#elif _M_ARM64 || _M_ARM64EC
    return _ReadStatusRegister(ARM64_PMCCNTR_EL0);
#elif __aarch64__
    return __builtin_arm_rsr64("cntvct_el0");
#elif __powerpc64__
    return __builtin_ppc_get_timebase();
#else
    return std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

enum class DeviationFilter {
    None,
    Sigma,
    MAD,
};

struct Options {
    double max_time = 0.5;
    DeviationFilter deviation_filter = DeviationFilter::MAD;
};

struct State {
private:
    friend struct Reporter;

    int64_t t0 = 0;
    int64_t time_elapsed = 0;
    int64_t max_time = 1;
    struct Chunk {
        static const size_t kMaxPerChunk = 65536;
        size_t count = 0;
        Chunk *next = nullptr;
        int64_t records[kMaxPerChunk]{};
    };
    int64_t iteration_count = 0;
    Chunk *rec_chunks = new Chunk();
    Chunk *rec_chunks_tail = rec_chunks;
    int64_t pause_t0 = 0;
    int64_t const *args = nullptr;
    size_t nargs = 0;
    int64_t items_processed = 0;
    DeviationFilter deviation_filter = DeviationFilter::None;

public:
    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE int64_t arg(size_t i) const {
        if (i > nargs)
            return 0;
        return args[i];
    }

    State(Options const &options) {
        set_max_time(options.max_time);
        set_deviation_filter(options.deviation_filter);
    }

    ~State() {
        Chunk *current = rec_chunks;
        while (current != nullptr) {
            Chunk *next = current->next;
            delete current;
            current = next;
        }
    }

    State(State &&) = delete;
    State &operator=(State &&) = delete;

    struct iterator {
    private:
        State &state;
        bool ok;

    public:
        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE iterator(State &state_, bool ok_) : state(state_), ok(ok_) {
            if (ok)
                state.start();
        }

        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE iterator &operator++() {
            state.stop();
            ok = state.next();
            if (ok)
                state.start();
            return *this;
        }

        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE iterator operator++(int) {
            iterator tmp = *this;
            ++*this;
            return tmp;
        }

        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE int operator*() const noexcept {
            return 0;
        }

        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE bool operator!=(iterator const &that) const noexcept {
            return ok != that.ok;
        }

        TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE bool operator==(iterator const &that) const noexcept {
            return ok == that.ok;
        }
    };

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE iterator begin() {
        return iterator{*this, true};
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE iterator end() {
        return iterator{*this, false};
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void start() {
        sfence();
        t0 = now();
        lfence();
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void pause() {
        pause_t0 = now();
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void resume() {
        int64_t t1 = now();
        t0 -= t1 - pause_t0;
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void stop() {
        mfence();
        stop(now());
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void start(int64_t t) {
        t0 = t;
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE void stop(int64_t t) {
        int64_t dt = t - t0;
        time_elapsed += dt;
        auto &chunk = *rec_chunks_tail;
        chunk.records[chunk.count++] = dt;
        if (chunk.count == chunk.kMaxPerChunk) {
            Chunk *new_node = new Chunk();
            rec_chunks_tail->next = new_node;
            rec_chunks_tail = new_node;
        }
        ++iteration_count;
    }

    TINYBENCH_ALWAYS_INLINE TINYBENCH_OPTIMIZE bool next() {
        bool ok = time_elapsed <= max_time;
#if __GNUC__
        return __builtin_expect(ok, 1);
#else
        return ok;
#endif
    }

    int64_t iterations() const noexcept {
        return iteration_count;
    }

    int64_t times() const noexcept {
        return time_elapsed;
    }

    void set_max_time(double t) {
        max_time = (int64_t)(t * 1000000000);
    }

    void set_deviation_filter(DeviationFilter f) {
        deviation_filter = f;
    }

    void set_items_processed(int64_t num) {
        items_processed = num;
    }
};

struct Entry {
    void (*func)(State &);
    const char *name;
    std::vector<std::vector<int64_t>> args{};
};

int register_entry(Entry ent);

struct Reporter {
    struct Row {
        double med;
        double avg;
        double stddev;
        double min;
        double max;
        int64_t count;
    };

    void run_entry(Entry const &ent, Options const &options = {});
    void run_all(Options const &options = {});

    virtual void report_state(const char *name, State &state);
    virtual void write_report(const char *name, Row const &row) = 0;

    virtual ~Reporter() = default;
};

Reporter *makeConsoleReporter();
Reporter *makeCSVReporter(const char *path);
Reporter *makeSVGReporter(const char *path);
Reporter *makeNullReporter();
Reporter *makeMultipleReporter(std::vector<Reporter *> const &reporters);

void _do_not_optimize_impl(void *p);

template <class T>
TINYBENCH_OPTIMIZE void do_not_optimize(T &&t) {
#if __GNUC__
    asm volatile ("" : "+m" (t) :: "memory");
#else
    _do_not_optimize_impl(std::addressof(t));
#endif
}

#define BENCHMARK_DEFINE(name) \
static int _defbench_##name = ::tinybench::register_entry({name, #name});
#define BENCHMARK(name, ...) \
extern "C" void name(::tinybench::State &); \
static int _defbench_##name = ::tinybench::register_entry({name, #name, __VA_ARGS__}); \
extern "C" TINYBENCH_NOINLINE void name(::tinybench::State &h)

std::vector<int64_t> linear_range(int64_t begin, int64_t end, int64_t step = 1);
std::vector<int64_t> log_range(int64_t begin, int64_t end, double factor = 2);

}

#ifdef TINYBENCH_IMPL
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#if __linux__
#include <fcntl.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>
#elif __APPLE__
#include <mach/mach_time.h>
#elif _WIN32
#include <windows.h>
#endif

namespace tinybench {

namespace {

std::vector<Entry> &entries() {
    static std::vector<Entry> instance;
    return instance;
}

int64_t get_cpu_freq() {
#if __linux__
    int fd;
    int result;
    fd = open("/proc/cpuinfo", O_RDONLY);
    if (fd != -1) {
        char buf[4096];
        ssize_t n;
        n = read(fd, buf, sizeof buf);
        if (__builtin_expect(n, 1) > 0) {
            char *mhz = (char *)memmem(buf, n, "cpu MHz", 7);
            if (mhz != NULL) {
                char *endp = buf + n;
                int seen_decpoint = 0;
                int ndigits = 0;
                while (mhz < endp && (*mhz < '0' || *mhz > '9') && *mhz != '\n')
                    ++mhz;
                while (mhz < endp && *mhz != '\n') {
                    if (*mhz >= '0' && *mhz <= '9') {
                        result *= 10;
                        result += *mhz - '0';
                        if (seen_decpoint)
                            ++ndigits;
                    } else if (*mhz == '.')
                        seen_decpoint = 1;

                    ++mhz;
                }
                while (ndigits++ < 6)
                    result *= 10;
            }
        }

        close(fd);
    }

    return result;
#elif __APPLE__
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return info.denom * 1000000000ULL / info.numer;
#elif _WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return freq.QuadPart;
#endif
}

void setup_affinity() {
    unsigned int cpu = 0;
#if __linux__
    getcpu(&cpu, nullptr);
#elif _WIN32
    cpu = GetCurrentProcessorNumber();
#endif
#if __linux__
    std::string path = "/sys/devices/system/cpu/cpu";
    path += std::to_string(cpu);
    path += "/cpufreq/scaling_governor";
    FILE *fp = fopen(path.c_str(), "r");
    if (fp) {
        char buf[64];
        fgets(buf, sizeof(buf), fp);
        fclose(fp);
        if (strncmp(buf, "performance", 11)) {
            fprintf(stderr, "\033[33;1mWARNING: CPU scaling detected! Run this to disable:\n"
                    "sudo cpupower frequency-set --governor performance\n\033[0m");
            fp = fopen(path.c_str(), "w");
            if (fp) {
                fputs("performance", fp);
                fclose(fp);
            }
        }
    }
#endif
#if __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(gettid(), sizeof(cpuset), &cpuset);
    struct sched_param param;
    memset(&param, 0, sizeof(param));
    param.sched_priority = sched_get_priority_max(SCHED_BATCH);
    sched_setscheduler(gettid(), SCHED_BATCH, &param);
#elif _WIN32
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
}

template <class T>
T find_median(T *begin, size_t n) {
    if (n % 2 == 0) {
        std::nth_element(begin, begin + n / 2, begin + n);
        std::nth_element(begin, begin + (n - 1) / 2, begin + n);
        return (begin[(n - 1) / 2] + begin[n / 2]) / 2;
    } else {
        std::nth_element(begin, begin + n / 2, begin + n);
        return begin[n / 2];
    }
}

}

int register_entry(Entry ent) {
    entries().push_back(ent);
    return 1;
}

void Reporter::run_entry(Entry const &ent, Options const &options) {
    size_t nargs = ent.args.size();
    if (nargs != 0) {
        std::vector<size_t> indices(nargs, 0);
        bool done;
        do {
            State state(options);

            std::vector<int64_t> args(nargs);
            state.args = args.data();
            state.nargs = nargs;

            std::string new_name = ent.name;
            for (size_t i = 0; i < nargs; i++) {
                int64_t value = ent.args[i][indices[i]];
                args[i] = value;
                new_name += '/';
                if (value == 0) {
                    new_name += '0';
                } else if (value % (1024 * 1024 * 1024) == 0) {
                    new_name += std::to_string(value / (1024 * 1024 * 1024)) + 'G';
                } else if (value % (1024 * 1024) == 0) {
                    new_name += std::to_string(value / (1024 * 1024)) + 'M';
                } else if (value % 1024 == 0) {
                    new_name += std::to_string(value / 1024) + 'k';
                } else {
                    new_name += std::to_string(value);
                }
            }

            ent.func(state);
            report_state(new_name.c_str(), state);

            done = true;
            for (size_t i = 0; i < nargs; i++) {
                ++indices[i];
                if (indices[i] >= ent.args[i].size()) {
                    indices[i] = 0;
                    continue;
                } else {
                    done = false;
                    break;
                }
            }
        } while (!done);

    } else {
        State state(options);
        state.max_time = (int64_t)(1000000000 * options.max_time);

        ent.func(state);
        report_state(ent.name, state);
    }
}

void Reporter::report_state(const char *name, State &state) {
    int64_t count = 0;
    int64_t max = INT64_MIN;
    int64_t min = INT64_MAX;
    double sum = 0;
    double square_sum = 0;

    std::vector<int64_t> records;
    for (State::Chunk *chunk = state.rec_chunks; chunk; chunk = chunk->next) {
        for (size_t i = 0; i < chunk->count; ++i) {
            records.push_back(chunk->records[i]);
        }
    }
    for (auto const &x: records) {
        sum += x;
        square_sum += x * x;
        max = std::max(x, max);
        min = std::min(x, min);
        ++count;
    }

    double avg = sum / count;
    double square_avg = square_sum / count;
    double stddev = std::sqrt(square_avg - avg * avg);

    if (state.deviation_filter != DeviationFilter::None) {
        sum = 0;
        square_sum = 0;
        count = 0;
        max = INT64_MIN;
        min = INT64_MAX;

        size_t nrecs = records.size();
        std::vector<bool> ok(nrecs);

        switch (state.deviation_filter) {
        case DeviationFilter::None:
            break;
        case DeviationFilter::MAD:
            {
                int64_t median = find_median(records.data(), records.size());
                std::vector<int64_t> deviations(nrecs);
                auto dev = deviations.data();
                for (auto const &x: records) {
                    *dev++ = std::abs(x - median);
                }
                int64_t mad = find_median(deviations.data(), deviations.size());
                auto okit = ok.begin();
                for (auto const &x: records) {
                    *okit++ = std::abs(x - median) <= 12 * mad;
                }
            }
            break;

        case DeviationFilter::Sigma:
            {
                auto okit = ok.begin();
                for (auto const &x: records) {
                    *okit++ = std::abs(x - avg) <= 3 * stddev;
                }
            }
            break;
        }

        auto okit = ok.begin();
        for (auto const &x: records) {
            if (*okit++) {
                sum += x;
                square_sum += x * x;
                max = std::max(x, max);
                min = std::min(x, min);
                ++count;
            }
        }

        avg = sum / count;
        square_avg = square_sum / count;
        stddev = std::sqrt(square_avg - avg * avg);
    }

#if __x86_64__ || _M_AMD64
#if __GNUC__
    const int64_t kFixedOverhead = 44;
#else
    const int64_t kFixedOverhead = 52;
#endif
#else
    const int64_t kFixedOverhead = 0;
#endif

    int64_t med = find_median(records.data(), records.size());
    med -= kFixedOverhead;
    avg -= kFixedOverhead;
    min -= kFixedOverhead;
    max -= kFixedOverhead;

    double rate = state.items_processed ?
        (double)state.iteration_count / state.items_processed
        : 1.0;
    write_report(name, Reporter::Row{
        med * rate, avg * rate, stddev * rate,
        min * rate, max * rate, count,
    });
}

void Reporter::run_all(Options const &options) {
    setup_affinity();
    for (Entry const &ent: entries()) {
        run_entry(ent, options);
    }
}

void _do_not_optimize_impl(void *p) {
    (void)p;
}

std::vector<int64_t> linear_range(int64_t begin, int64_t end, int64_t step) {
    std::vector<int64_t> ret;
    for (int64_t i = begin; i <= end; i += step) {
        ret.push_back(i);
    }
    return ret;
}

std::vector<int64_t> log_range(int64_t begin, int64_t end, double factor) {
    std::vector<int64_t> ret;
    if (factor >= 1) {
        int64_t last_i = begin - 1;
        for (double d = begin; d <= end; d *= factor) {
            int64_t i = int64_t(d);
            if (last_i != i)
                ret.push_back(i);
            i = last_i;
        }
    }
    return ret;
}

namespace {

int guess_prec(int max, double x, double mag = 10.0) {
    int n = 0;
    while (n < max - 2 && x < mag)
        x *= 10, ++n;
    return n;
}

const char *fit_order(double &x) {
    if (x >= 1024 * 1024 * 1024) {
        x /= 1024 * 1024 * 1024;
        return "G";
    } else if (x >= 1024 * 1024) {
        x /= 1024 * 1024;
        return "M";
    } else if (x >= 1024) {
        x /= 1024;
        return "k";
    } else {
        return "";
    }
}

struct ConsoleReporter : Reporter {
    ConsoleReporter() {
        printf("%26s %11s %11s %6s %9s\n", "name", "med", "avg", "std", "n");
        printf("-------------------------------------------------------------------\n");
    }

    void write_report(const char *name, Reporter::Row const &row) override {
        printf("%26s %11.*lf %11.*lf %6.*lf %9ld\n",
               name, guess_prec(11, row.med), row.med, guess_prec(11, row.avg), row.avg, guess_prec(6, row.stddev), row.stddev, row.count);
    }
};

struct CSVReporter : Reporter {
    FILE *fp;

    CSVReporter(const char *filename) {
        fp = fopen(filename, "w");
        if (!fp)
            abort();
        fprintf(fp, "name,avg,std,min,max,n\n");
    }

    CSVReporter(CSVReporter &&) = delete;

    ~CSVReporter() {
        fclose(fp);
    }

    void write_report(const char *name, Reporter::Row const &row) override {
        fprintf(fp, "%s,%lf,%lf,%lf,%lf,%ld\n",
               name, row.avg, row.stddev, row.min, row.max, row.count);
    }
};

struct SVGReporter : Reporter {
    FILE *fp;

    struct Bar {
        std::string name;
        double value;
        double height;
        double delta_up;
        double delta_mid;
        double delta_down;
        double stddev_max;
        double stddev_min;
    };

    std::vector<Bar> bars;

    SVGReporter(const char *filename) {
        fp = fopen(filename, "w");
        if (!fp)
            abort();
    }

    SVGReporter(SVGReporter &&) = delete;

    ~SVGReporter() {
        double w = 1920;
        double h = 1080;
        fprintf(fp, "<svg viewBox=\"0 0 %lf %lf\" xmlns=\"http://www.w3.org/2000/svg\">\n", w, h);
        fprintf(fp, "<style type=\"text/css\">\n"
                    ".bar {\n"
                    "  stroke: #000000;\n"
                    "  fill: #779977;\n"
                    "}\n"
                    ".tip {\n"
                    "  stroke: #223344;\n"
                    "  fill: none;\n"
                    "}\n"
                    ".stddev {\n"
                    "  stroke: none;\n"
                    "  fill: #223344;\n"
                    "  opacity: 0.25;\n"
                    "}\n"
                    ".label {\n"
                    "  font-family: monospace;\n"
                    "  color: #000000;\n"
                    "  dominant-baseline: central;\n"
                    "  text-anchor: middle;\n"
                    "}\n"
                    ".value {\n"
                    "  font-family: monospace;\n"
                    "  color: #000000;\n"
                    "  dominant-baseline: central;\n"
                    "  text-anchor: middle;\n"
                    "}\n"
                    "</style>\n");
        fprintf(fp, "<rect x=\"0\" y=\"0\" width=\"%lf\" height=\"%lf\" fill=\"lightgray\" />\n", w, h);

        double xscale = (w - 80) / (bars.size() + 1);
        double ymax = 0;
        for (size_t i = 0; i < bars.size(); i++) {
            ymax = std::max(ymax, bars[i].height + std::min(bars[i].delta_up, bars[i].height));
        }
        double yscale = (h - 120) / ymax;
        for (size_t i = 0; i < bars.size(); i++) {
            double bar_width = 0.65 * xscale;
            double x = 40 + (i + 1) * xscale;
            double y = h - 60;
            double bar_height = bars[i].height * yscale;
            double avg_width = 0.35 * xscale;
            double tip_width = 0.15 * xscale;
            double tip_height_up = bars[i].delta_up * yscale;
            double tip_height_mid = bars[i].delta_mid * yscale;
            double tip_height_down = bars[i].delta_down * yscale;
            fprintf(fp, "<rect class=\"bar\" x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" />\n",
                   x - bar_width * 0.5, y - bar_height, bar_width, bar_height);
            fprintf(fp, "<rect class=\"stddev\" x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" />\n",
                    x - avg_width * 0.5, y - bars[i].stddev_max * yscale, avg_width,
                    (bars[i].stddev_max - bars[i].stddev_min) * yscale);
            fprintf(fp, "<line class=\"tip\" x1=\"%lf\" y1=\"%lf\" x2=\"%lf\" y2=\"%lf\" />\n",
                   x, y - bar_height - tip_height_up, x, y - bar_height - tip_height_down);
            fprintf(fp, "<line class=\"tip\" x1=\"%lf\" y1=\"%lf\" x2=\"%lf\" y2=\"%lf\" />\n",
                   x - tip_width * 0.5, y - bar_height - tip_height_up, x + tip_width * 0.5, y - bar_height - tip_height_up);
            fprintf(fp, "<line class=\"tip\" x1=\"%lf\" y1=\"%lf\" x2=\"%lf\" y2=\"%lf\" />\n",
                   x - tip_width * 0.5, y - bar_height - tip_height_mid, x + tip_width * 0.5, y - bar_height - tip_height_mid);
            fprintf(fp, "<line class=\"tip\" x1=\"%lf\" y1=\"%lf\" x2=\"%lf\" y2=\"%lf\" />\n",
                   x - tip_width * 0.5, y - bar_height - tip_height_down, x + tip_width * 0.5, y - bar_height - tip_height_down);
            double value = bars[i].value;
            const char *order = fit_order(value);
            fprintf(fp, "<text class=\"value\" x=\"%lf\" y=\"%lf\">%.*lf%s</text>\n",
                   x, y - bar_height - 20, guess_prec(11, value), value, order);
            fprintf(fp, "<text class=\"label\" x=\"%lf\" y=\"%lf\">%s</text>\n",
                   x, h - 30, bars[i].name.c_str());
        }
        fprintf(fp, "</svg>\n");
        fclose(fp);
    }

    void write_report(const char *name, Reporter::Row const &row) override {
        auto axis_scale = [] (double x) {
            if (x <= 1)
                return x;
            return std::log(x) + 1;
        };
        auto height = axis_scale(row.med);
        auto height_up = axis_scale(row.max);
        auto height_mid = axis_scale(row.med);
        auto height_down = axis_scale(row.min);
        auto stddev_up = axis_scale(row.avg + row.stddev);
        auto stddev_down = axis_scale(row.avg - row.stddev);
        bars.push_back({
            name,
            row.avg,
            height,
            height_up - height,
            height_mid - height,
            height_down - height,
            stddev_up,
            stddev_down,
        });
    }
};

struct NullReporter : Reporter {
    void write_report(const char *name, Reporter::Row const &row) override {
        (void)name;
        (void)row;
    }
};

struct MultipleReporter : Reporter {
    std::vector<std::unique_ptr<Reporter>> reporters;

    MultipleReporter(std::vector<Reporter *> const &rs) {
        for (auto *r: rs) {
            reporters.emplace_back(r);
        }
    }

    void write_report(const char *name, Reporter::Row const &row) override {
        for (auto &r: reporters) {
            r->write_report(name, row);
        }
    }
};

}

Reporter *makeConsoleReporter() {
    return new ConsoleReporter();
}

Reporter *makeCSVReporter(const char *path) {
    return new CSVReporter(path);
}

Reporter *makeSVGReporter(const char *path) {
    return new SVGReporter(path);
}

Reporter *makeNullReporter() {
    return new NullReporter();
}

Reporter *makeMultipleReporter(std::vector<Reporter *> const &reporters) {
    return new MultipleReporter(reporters);
}

}
#endif
