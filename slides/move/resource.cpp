#include "resource.hpp"
#include <cstdio>

struct Resource::Self {
    FILE *p;

    Self() {
        puts("打开文件");
        p = fopen("CMakeCache.txt", "r");
    }

    Self(Self &&) = delete;

    void speak() {
        printf("使用文件 %p\n", p);
    }

    ~Self() {
        puts("关闭文件");
        fclose(p);
    }
};

Resource::Resource() : self(std::make_unique<Self>()) {
}

void Resource::speak() {
    return self->speak();
}

Resource::~Resource() = default;
