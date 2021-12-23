#include <cstdio>
#include <memory>

struct MyClassWithVeryLongName {
};

auto func() {
    return std::make_shared<MyClassWithVeryLongName>();
}

int main() {
    auto p = func();
}
