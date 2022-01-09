#pragma once

#include <memory>

struct Course {
    struct Impl;
    std::shared_ptr<Impl> impl;

    Course();
    ~Course();
    void func() const;
};
