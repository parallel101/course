#pragma once

#include <barrier>
#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __linux__
# include <sched.h>
# include <unistd.h>
#endif
#if defined(__unix__) && defined(__has_include)
# if __has_include(<cxxabi.h>)
#  include <cxxabi.h>
# endif
#endif

template <std::size_t I>
struct MTIndex {
    explicit MTIndex() = default;

    static inline constexpr std::size_t index = I;
};

struct MTTest {
private:
    virtual void setupThreadLocal(std::size_t index) noexcept = 0;
    virtual void resetTest() noexcept = 0;
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
            (void)((Is == index
                        ? ((testFunc =
                                [] {
                                    static_cast<TestClass *>(testClass)->entry(
                                        MTIndex<Is>());
                                }),
                           true)
                        : false) ||
                   ...);
            if (!testFunc) {
                std::terminate();
            }
        }

        void setupThreadLocal(std::size_t index) noexcept override {
            setupThreadLocalImpl(
                index,
                std::make_index_sequence<CountEntries<TestClass>::value>());
        }

        void resetTest() noexcept override {
            m_testClass.~TestClass();
            new (&m_testClass) TestClass();
        }

        std::size_t numEntries() const noexcept override {
            return CountEntries<TestClass>::value;
        }

        std::string testName() const noexcept override {
            std::string name = typeid(TestClass).name();
#if defined(__unix__) && defined(__has_include)
# if __has_include(<cxxabi.h>)
            char *demangled =
                abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr);
            name = demangled;
            std::free(demangled);
# endif
#endif
            return name;
        }

        alignas(64) TestClass m_testClass;
    };

    struct MTTestPool final {
        MTTestPool(std::unique_ptr<MTTest> testData, std::size_t repeats)
            : m_threads(testData->numEntries()),
              m_barrierBegin(testData->numEntries()),
              m_barrierEnd(testData->numEntries() + 1),
              m_repeats(repeats),
              m_testData(std::move(testData)) {
            std::mt19937 rng(
                std::chrono::steady_clock::now().time_since_epoch().count());
            std::size_t maxCores = std::thread::hardware_concurrency();
            std::uniform_int_distribution<std::size_t> uni(0, maxCores - 1);
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
            std::cout << "在测试 " << m_testData->testName() << " 中:\n";
            for (auto &&[k, v]: m_statistics) {
                std::cout << "  " << k << " 出现了 " << v << " 次\n";
            }
        }

    private:
        void testThread(std::size_t index, std::size_t cpuid) noexcept {
            // barrier 是为了保证所有线程尽可能同时开始运行，不要一前一后
            // sched_setaffinity 是为了绑定线程到不同的 CPU
            // 核心上，保证线程之间是并行而不是并发
#ifdef __linux__
            cpu_set_t cpu;
            CPU_ZERO(&cpu);
            CPU_SET(cpuid, &cpu);
            sched_setaffinity(gettid(), sizeof(cpu),
                              &cpu); // 绑定线程到固定核心
#endif
            m_testData->setupThreadLocal(index);
            for (int t = 0; t < 10; ++t) {
                m_barrierBegin.arrive_and_wait(); // 等待其他线程就绪
            }
            for (std::size_t i = 0; i < m_repeats; ++i) {
                m_barrierBegin.arrive_and_wait(); // 等待其他线程就绪
                m_barrierBegin.arrive_and_wait(); // 等两次避免同步不完全
                testFunc();
                m_barrierEnd.arrive_and_wait(); // 等待统计线程开始
                m_barrierEnd.arrive_and_wait(); // 等待统计线程结束
            }
        }

        void statisticThread(std::size_t cpuid) {
#ifdef __linux__
            cpu_set_t cpu;
            CPU_ZERO(&cpu);
            CPU_SET(cpuid, &cpu);
            sched_setaffinity(gettid(), sizeof(cpu),
                              &cpu); // 绑定线程到固定核心
#endif
            for (std::size_t i = 0; i < m_repeats; ++i) {
                m_barrierEnd.arrive_and_wait(); // 等待其他线程结束
                ++m_statistics[result];         // 将当前结果纳入统计
                m_testData->resetTest();        // 重置所有状态
                result = Result();              // 重置结果
                m_barrierEnd.arrive_and_wait(); // 通知其他线程开始
            }
        }

        std::vector<std::thread> m_threads;
        std::thread m_statThread;
        std::barrier<> m_barrierBegin;
        std::barrier<> m_barrierEnd;
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
