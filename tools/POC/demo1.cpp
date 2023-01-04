#include <benchmark/benchmark.h>
#include <cstdint>
#include <thread>
#include <mutex>
#include <array>
#include <atomic>
#include <numeric>
#include <stdexcept>
#include <string>

static constexpr size_t kNumBuckets = 32;
static constexpr size_t kNumThreads = 128;
static constexpr size_t kNumLoops = 65536;

struct LockCounter {
    std::mutex mtx;
    uint32_t count = 0;

    void add(uint32_t n) {
        std::unique_lock lck(mtx);
        count += n;
    }

    uint32_t get() const {
        return count;
    }
};

struct AtomicCounter {
    std::atomic<uint32_t> count = 0;

    void add(uint32_t n) {
        count += n;
    }

    uint32_t get() const {
        return count;
    }
};

struct LockArrayCounter {
    struct Bucket {
        std::mutex mtx;
        uint32_t count = 0;
    };
    std::array<Bucket, kNumBuckets> buckets;

    void add(uint32_t n) {
        size_t bucketId = std::hash<std::thread::id>()(std::this_thread::get_id()) % kNumBuckets;
        Bucket &bucket = buckets[bucketId];
        std::unique_lock lck(bucket.mtx);
        bucket.count += n;
    }

    uint32_t get() const {
        return std::accumulate(buckets.begin(), buckets.end(),
                               uint32_t(0), [] (uint32_t acc, Bucket const &bucket) {
                                   return acc + bucket.count;
                               });
    }
};

struct LockPaddedCounter {
    struct Bucket {
        std::mutex mtx;
        uint32_t count = 0;
        char padding[256];
    };
    std::array<Bucket, kNumBuckets> buckets;

    void add(uint32_t n) {
        uint32_t bucketId = std::hash<std::thread::id>()(std::this_thread::get_id());
        Bucket &bucket = buckets[bucketId % kNumBuckets];
        std::unique_lock lck(bucket.mtx);
        bucket.count += n;
    }

    uint32_t get() const {
        return std::accumulate(buckets.begin(), buckets.end(),
                               uint32_t(0), [] (uint32_t acc, Bucket const &bucket) {
                                   return acc + bucket.count;
                               });
    }
};

struct AtomicArrayCounter {
    struct Bucket {
        std::atomic<uint32_t> count = 0;
    };
    std::array<Bucket, kNumBuckets> buckets;

    void add(uint32_t n) {
        uint32_t bucketId = std::hash<std::thread::id>()(std::this_thread::get_id());
        Bucket &bucket = buckets[bucketId % kNumBuckets];
        bucket.count += n;
    }

    uint32_t get() const {
        return std::accumulate(buckets.begin(), buckets.end(),
                               uint32_t(0), [] (uint32_t acc, Bucket const &bucket) {
                                   return acc + bucket.count;
                               });
    }
};

struct AtomicPaddedCounter {
    struct Bucket {
        std::atomic<uint32_t> count = 0;
        char padding[256];
    };
    std::array<Bucket, kNumBuckets> buckets;

    void add(uint32_t n) {
        uint32_t bucketId = std::hash<std::thread::id>()(std::this_thread::get_id());
        Bucket &bucket = buckets[bucketId % kNumBuckets];
        bucket.count += n;
    }

    uint32_t get() const {
        return std::accumulate(buckets.begin(), buckets.end(),
                               uint32_t(0), [] (uint32_t acc, Bucket const &bucket) {
                                   return acc + bucket.count;
                               });
    }
};

template <class LockCounterT>
static void testCounter(benchmark::State& state) {
    for (auto _ : state) {
        LockCounterT ctr;
        std::array<std::thread, kNumThreads> threads;
        for (size_t i = 0; i < kNumThreads; i++) {
            threads[i] = std::thread([&ctr] {
                for (size_t i = 0; i < kNumLoops; i++) {
                    ctr.add(1);
                }
            });
        }
        for (size_t i = 0; i < kNumThreads; i++) {
            threads[i].join();
        }
        int res = ctr.get();
        benchmark::DoNotOptimize(res);
        if (res != kNumThreads * kNumLoops) {
            throw std::runtime_error("wrong result: " + std::to_string(res));
        }
    }
}
BENCHMARK_TEMPLATE(testCounter, LockCounter);
BENCHMARK_TEMPLATE(testCounter, AtomicCounter);
BENCHMARK_TEMPLATE(testCounter, LockArrayCounter);
BENCHMARK_TEMPLATE(testCounter, LockPaddedCounter);
BENCHMARK_TEMPLATE(testCounter, AtomicArrayCounter);
BENCHMARK_TEMPLATE(testCounter, AtomicPaddedCounter);

BENCHMARK_MAIN();

/* my result on 12 thread 6 core CPU:
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
testCounter<LockCounter>          727206561 ns      3571387 ns           10
testCounter<AtomicCounter>        133500534 ns      4680311 ns          100
testCounter<LockArrayCounter>     115707644 ns      2925223 ns          100
testCounter<LockPaddedCounter>     66132445 ns      3500832 ns          100
testCounter<AtomicArrayCounter>   109673523 ns      3206200 ns          100
testCounter<AtomicPaddedCounter>   20806265 ns      3462085 ns          206
*/
