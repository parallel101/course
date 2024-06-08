#pragma once

#include <cstdio>

struct C {
    C() {
        puts("C()");
    }

    C(C &&) noexcept {
        puts("C(C &&)");
    }

    C(C const &) {
        puts("C(C const &)");
    }

    C &operator=(C &&) noexcept {
        puts("operator=(C &&)");
        return *this;
    }

    C &operator=(C const &) {
        puts("operator=(C const &)");
        return *this;
    }

    ~C() {
        puts("~C()");
    }
};
