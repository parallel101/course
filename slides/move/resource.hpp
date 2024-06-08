#pragma once

#include <memory>

// P-IMPL 模式
struct Resource {
private:
    struct Self;

    std::unique_ptr<Self> self;

public:
    Resource();
    void speak();
    ~Resource();
};
