#pragma once

#include <iostream>
#include "print.h"

#define ASSERT(x, ...) do { if (!(x)) { this_test.mark_failed(); this_test.assert_failure(#x); this_test.assert_show_where(__VA_ARGS__); } } while (0)
#define ASSERT_T(x) ASSERT(x, this_test.assert_make_pair(#x, x))
#define ASSERT_F(x) ASSERT(!x, this_test.assert_make_pair(#x, x))
#define ASSERT_PRED(f, x) ASSERT(f(x), this_test.assert_make_pair(#x, x))
#define ASSERT_EQ(x, y) ASSERT(x == y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))
#define ASSERT_NE(x, y) ASSERT(x != y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))
#define ASSERT_LT(x, y) ASSERT(x < y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))
#define ASSERT_GT(x, y) ASSERT(x > y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))
#define ASSERT_LE(x, y) ASSERT(x <= y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))
#define ASSERT_GE(x, y) ASSERT(x >= y, this_test.assert_make_pair(#x, x), this_test.assert_make_pair(#y, y))

struct test_case_init {
    test_case_init(const char *name) : name(name) {
        std::cout << "------ " << name << '\n';
    }

    ~test_case_init() {
        if (!failed) {
            std::cout << "[ OK ] " << name << '\n';
        } else {
            std::cout << "[FAIL] " << name << '\n';
        }
    }

    test_case_init(test_case_init &&) = delete;

    void mark_failed() {
        failed = true;
    }

    operator bool() const {
        return true;
    }

    static void assert_failure(const char *condStr) {
        std::cout << "ASSERT_FAILURE:\n";
        std::cout << '\t' << condStr << '\n';
    }

    template <class T>
    static std::pair<const char *, T const *> assert_make_pair(const char *msg, T const &val) {
        return {msg, &val};
    }

    template <class ...Args>
    static void assert_show_where(Args ...args) {
        if constexpr (sizeof...(args)) {
            std::cout << "WHERE:\n";
            ((std::cout << '\t' << args.first << " = "
              << print_adaptor(*args.second) << '\n'),
             ...);
        }
    }

    const char *name = nullptr;
    bool failed = false;
};

#define TEST_CASE(name) if (auto this_test = test_case_init(#name))
