#pragma once

struct C {
    int *p;

    C() : p(new int) {
    }

    C(C const &) = delete;
    C &operator=(C const &) = delete;

    ~C() {
        delete p;
    }
};
