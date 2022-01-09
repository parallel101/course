#include "Course.h"

struct Course::Impl {
    int x, y, c, d, e;

    void func() {
        x += 1;
    }
};

Course::Course() : impl(std::make_unique<Impl>()) {}
Course::~Course() = default;
void Course::func() const { impl->func(); }
