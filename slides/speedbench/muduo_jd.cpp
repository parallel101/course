#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

size_t const batch = 64 * 1024;
size_t const warmup = 16;
size_t const repeats = 512;

// our method:
struct YMDDate {
    std::chrono::year year;
    std::chrono::month month;
    std::chrono::day day;
};

struct jdate_clock {
    using rep = int32_t;
    using period = std::chrono::days::period;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<jdate_clock>;

    static_assert(sizeof(rep) * 8 >= 32, "julian day must be at least 32-bit");

    static constexpr bool is_steady = false;

    static constexpr duration jdiff() {
        using namespace std::chrono;
        return duration_cast<duration>(sys_days{January / 1 / 1970} - (sys_days{November / 24 / -4713} - 12h));
    }

    static time_point now() noexcept {
        return from_system(std::chrono::system_clock::now());
    }

    static time_point from_system(std::chrono::system_clock::time_point sys_time) noexcept {
        return time_point{duration_cast<duration>(sys_time.time_since_epoch()) + jdiff()};
    }

    static std::chrono::system_clock::time_point to_system(time_point jdate_time) noexcept {
        return std::chrono::system_clock::time_point{duration_cast<std::chrono::system_clock::duration>(jdate_time.time_since_epoch() - jdiff())};
    }
};

int32_t toJulianDay(YMDDate const &date) {
    using namespace std::chrono;
    return jdate_clock::from_system(sys_days{date.year / date.month / date.day}).time_since_epoch().count();
}

// muduo method:
char require_32_bit_integer_at_least[sizeof(int) >= sizeof(int32_t) ? 1 : -1];

int getJulianDayNumber(int year, int month, int day) {
    (void)require_32_bit_integer_at_least; // no warning please
    int a = (14 - month) / 12;
    int y = year + 4800 - a;
    int m = month + 12 * a - 3;
    return day + (153 * m + 2) / 5 + y * 365 + y / 4 - y / 100 + y / 400 -
           32045;
}

// test and benchmark:
[[gnu::noinline]] void compute0(YMDDate const *date, int *__restrict result) {
    for (size_t i = 0; i < batch; ++i) {
        result[i] =
            getJulianDayNumber((int)date[i].year, (unsigned)date[i].month, (unsigned)date[i].day);
    }
}

[[gnu::noinline]] void compute1(YMDDate const *date, int *__restrict result) {
    for (size_t i = 0; i < batch; ++i) {
        result[i] = toJulianDay(date[i]);
    }
}

int main() {
    std::vector<int> result(batch);
    std::vector<int> result2(batch);
    std::vector<YMDDate> date(batch);
    std::mt19937 rng;
    std::generate(date.begin(), date.end(), [&rng]() {
        using namespace std::chrono;
        YMDDate date;
        date.year = year(std::uniform_int_distribution(1970, 2070)(rng));
        date.month = month(std::uniform_int_distribution(1, 12)(rng));
        date.day = day(std::uniform_int_distribution(1, date.month == February ? 28 : 30)(rng));
        return date;
    });
    // first verify result matches
    compute0(date.data(), result.data());
    compute1(date.data(), result2.data());
    for (size_t i = 0; i < batch; ++i) {
        if (result[i] != result2[i]) {
            std::cout << "error at " << i << ": " << result[i] << " != " << result2[i] << "\n";
            return 1;
        }
    }
    std::cout << "test ok\n";

    // then benchmark
    std::array computes{compute0, compute1};
    for (size_t mode = 0; mode < std::size(computes); ++mode) {
        for (size_t t = 0; t < warmup; ++t) {
            computes[mode](date.data(), result.data());
        }
        auto t0 = std::chrono::steady_clock::now();
        for (size_t t = 0; t < repeats; ++t) {
            computes[mode](date.data(), result.data());
        }
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "compute" << mode << ": "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         t1 - t0)
                         .count()
                  << "s\n";
    }
}
