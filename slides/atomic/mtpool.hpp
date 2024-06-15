#pragma once

#include <map>
#include <string>
#include <atomic>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(__unix__) && __has_include(<sched.h>) && __has_include(<unistd.h>)
# include <sched.h>
# include <unistd.h>
#elif defined(_WIN32) && __has_include(<windows.h>)
# include <windows.h>
#endif
#if defined(__unix__) &&  __has_include(<cxxabi.h>)
# include <cxxabi.h>
#endif

struct SpinBarrier {
    explicit SpinBarrier(std::size_t n) noexcept
        : m_top_waiting((std::uint32_t)n - 1),
          m_num_waiting(0),
          m_sync_flip(0) {}

    bool arrive_and_wait() noexcept {
        bool old_flip = m_sync_flip.load(std::memory_order_relaxed);
        if (m_num_waiting.fetch_add(1, std::memory_order_relaxed) == m_top_waiting) {
            m_num_waiting.store(0, std::memory_order_relaxed);
            m_sync_flip.store(!old_flip, std::memory_order_release);
#if __cpp_lib_atomic_wait
            m_sync_flip.notify_all();
#endif
            return true;
        } else {
#if __cpp_lib_atomic_wait
            std::uint32_t retries = 255;
            do {
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
                if (m_sync_flip.load(std::memory_order_acquire) != old_flip)
                    return false;
            } while (--retries);
#else
            while (m_sync_flip.load(std::memory_order_acquire) == step)
                ;
#endif
            m_sync_flip.wait(old_flip, std::memory_order_acquire);
            return false;
        }
    }

private:
    std::uint32_t const m_top_waiting;
    std::atomic<std::uint32_t> m_num_waiting;
    std::atomic<bool> m_sync_flip;
};

template <std::size_t I>
struct MTIndex {
    explicit MTIndex() = default;
};

template <std::size_t IBegin, std::size_t IEnd>
struct MTRangeIndex {
    explicit MTRangeIndex() = default;

    template <std::size_t I, std::enable_if_t<(I >= IBegin && I <= IEnd), int> = 0>
    MTRangeIndex(MTIndex<I>) noexcept {}
};

struct MTTest {
private:
    virtual void setupThreadLocal(std::size_t index) noexcept = 0;
    virtual void onSetup() noexcept = 0;
    virtual void onTeardown() noexcept = 0;
    virtual void resetState() noexcept = 0;
    virtual std::size_t numEntries() const noexcept = 0;
    virtual std::string testName() const noexcept = 0;

    static inline thread_local void (*testFunc)();
    static inline thread_local void *testClass;

public:
    using Result = int;

    static inline Result result;

    MTTest() = default;
    virtual ~MTTest() = default;

private:
    template <class TestClass, class = void>
    struct HasSetup : std::false_type {};

    template <class TestClass>
    struct HasSetup<
        TestClass, std::void_t<decltype(std::declval<TestClass>().setup())>>
        : std::true_type {};

    template <class TestClass, class = void>
    struct HasTeardown : std::false_type {};

    template <class TestClass>
    struct HasTeardown<
        TestClass, std::void_t<decltype(std::declval<TestClass>().teardown())>>
        : std::true_type {};

    template <class TestClass, std::size_t Begin = 0, class = void>
    struct CountEntries : std::integral_constant<std::size_t, Begin> {};

    template <class TestClass, std::size_t Begin>
    struct CountEntries<TestClass, Begin,
                        std::void_t<decltype(std::declval<TestClass>().entry(
                            MTIndex<Begin>()))>>
        : CountEntries<TestClass, Begin + 1> {};

    template <class TestClass, class MTTest = MTTest>
    struct MTTestImpl final : MTTest {
        MTTestImpl() noexcept : m_testClass() {}

        MTTestImpl(MTTestImpl &&) = delete;
        MTTestImpl(MTTestImpl const &) = delete;

    private:
        template <std::size_t... Is>
        void setupThreadLocalImpl(std::size_t index,
                                  std::index_sequence<Is...>) noexcept {
            testClass = &m_testClass;
            testFunc = nullptr;
            // 带 break 的静态 for 循环，第二步是通过 ... 搏杀暴徒
            (void)((Is == index
                        ? ((testFunc =
                                [] {
                                    static_cast<TestClass *>(testClass)->entry(
                                        MTIndex<Is>());
                                }),
                           true)
                        : false) ||
                   ...);
        }

        void setupThreadLocal(std::size_t index) noexcept override {
            // 带 break 的静态 for 循环需要分两步走，第一步是通过 make_index_sequence 领域展开
            setupThreadLocalImpl(
                index,
                std::make_index_sequence<CountEntries<TestClass>::value>());
        }

        void onSetup() noexcept override {
            if constexpr (HasSetup<TestClass>::value) {
                m_testClass.setup();
            }
        }

        void onTeardown() noexcept override {
            if constexpr (HasTeardown<TestClass>::value) {
                m_testClass.teardown();
            }
        }

        void resetState() noexcept override {
            m_testClass.~TestClass();
            new (&m_testClass) TestClass();
            onSetup();
        }

        std::size_t numEntries() const noexcept override {
            static_assert(CountEntries<TestClass>::value > 0);
            return CountEntries<TestClass>::value;
        }

        std::string testName() const noexcept override {
            std::string name = typeid(TestClass).name();
#if defined(__unix__) &&  __has_include(<cxxabi.h>)
            char *demangled =
                abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr);
            name = demangled;
            std::free(demangled);
#endif
            return name;
        }

        alignas(64) TestClass m_testClass;
    };

    static std::string resultToString(Result result) {
        return std::to_string(result);
    };

    struct MTTestPool final {
        MTTestPool(std::unique_ptr<MTTest> testData, std::size_t repeats)
            : m_threads(testData->numEntries()),
              m_barrier(testData->numEntries() + 1),
              m_repeats(repeats),
              m_testData(std::move(testData)) {
            std::size_t maxCores = std::thread::hardware_concurrency();
            for (std::size_t i = 0; i < m_threads.size(); ++i) {
                m_threads[i] =
                    std::thread(&MTTestPool::testThread, this, i, i % maxCores);
            }
            m_statThread = std::thread(&MTTestPool::statisticThread, this,
                                       m_threads.size() % maxCores);
        }

        void joinAll() {
            for (auto &&t: m_threads) {
                t.join();
            }
            m_statThread.join();
        }

        void showStatistics() {
            /* std::cout << "在测试 " << m_testData->testName() << " 中:\n"; */
            /* for (auto &&[result, count]: m_statistics) { */
            /*     std::cout << "  " << result << " 出现了 " << count << " 次\n"; */
            /* } */
            printf("在测试 %s 中:\n", m_testData->testName().c_str());
            for (auto &&[result, count]: m_statistics) {
                printf("  %s 出现了 %zu 次\n", resultToString(result).c_str(), count);
            }
        }

    private:
        void setThisThreadAffinity(std::size_t cpuid) {
             // 绑定当前线程到指定 CPU 核心
#if defined(__unix__) && __has_include(<sched.h>) && __has_include(<unistd.h>)
            cpu_set_t cpu;
            CPU_ZERO(&cpu);
            CPU_SET(cpuid, &cpu);
            sched_setaffinity(gettid(), sizeof(cpu),
                              &cpu);
#elif defined(_WIN32) && __has_include(<windows.h>)
            SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpuid);
#endif
        }

        void testThread(std::size_t index, std::size_t cpuid) noexcept {
            // barrier 是为了保证所有线程尽可能同时开始运行，不要一前一后
            // setAffinity 是为了绑定线程到不同的 CPU
            // 核心上，保证线程之间是并行而不是并发
            setThisThreadAffinity(cpuid);
            m_testData->setupThreadLocal(index);
            std::this_thread::yield();
            for (int t = 0; t < 8; ++t) {    // 热身运动
                m_barrier.arrive_and_wait(); // 等待其他线程就绪
            }
            for (std::size_t i = 0; i < m_repeats; ++i) {
                testFunc();
                m_barrier.arrive_and_wait(); // 等待统计线程开始
                m_barrier.arrive_and_wait(); // 等待统计线程结束
            }
        }

        void statisticThread(std::size_t cpuid) noexcept {
            setThisThreadAffinity(cpuid);
            m_testData->onSetup();           // 初始化状态
            for (int t = 0; t < 8; ++t) {    // 热身运动
                m_barrier.arrive_and_wait(); // 等待其他线程就绪
            }
            for (std::size_t i = 0; i < m_repeats; ++i) {
                m_barrier.arrive_and_wait();    // 等待其他线程结束
                m_testData->onTeardown();       // 最终处理
                ++m_statistics[result];         // 将当前结果纳入统计
                m_testData->resetState();       // 重置所有状态
                result = Result();              // 重置结果
                m_barrier.arrive_and_wait();    // 通知其他线程开始
            }
        }

        std::vector<std::thread> m_threads;
        std::thread m_statThread;
        SpinBarrier m_barrier;
        std::size_t const m_repeats;
        std::unique_ptr<MTTest> const m_testData;
        std::map<Result, std::size_t> m_statistics;
    };

public:
    template <class TestClass>
    static void runTest(std::size_t repeats = 100000) {
        MTTestPool pool(std::make_unique<MTTestImpl<TestClass>>(), repeats);
        pool.joinAll();
        pool.showStatistics();
    }
};
