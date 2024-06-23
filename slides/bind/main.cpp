#include <iostream>
#include <memory>
#include <string>
#include "Signal.hpp"

struct Foo {
    void on_input(int i, int j) const {
        std::cout << "Foo of age " << age << " got i=" << i << ", j=" << j << '\n';
    }

    int age = 14;

    ~Foo() {
        std::cout << "Foo destruct\n";
    }
};

struct Bar {
    void on_input(int i) const {
        std::cout << "Bar of age " << age << " got " << i << '\n';
    }

    void on_exit(std::string msg1, std::string msg2) const {
        std::cout << "Bar got exit event: " << msg1 << " " << msg2 << '\n';
    }

    int age = 42;
};

std::shared_ptr<Bar> gbar;
std::shared_ptr<void> gmine;

struct Input {
    void main_loop() {
        int i;
        while (std::cin >> i) {
            on_input.emit(i);
            if (i == 42)
                gbar = nullptr;
        }
        on_exit.emit("hello", "world");
    }

    pengsig::Signal<int> on_input;
    pengsig::Signal<std::string, std::string> on_exit;
};

void test(std::string msg1, std::string msg2) {
    std::cout << "main received exit event: " << msg1 << " " << msg2 << '\n';
}

void test(int msg1, std::string msg2) {
    std::cout << "main received exit event: " << msg1 << " " << msg2 << '\n';
}

struct Mine : std::enable_shared_from_this<Mine> { // Mine 必须从 make_shared 创建
    void register_on(Input &input) {
        input.on_input.connect(weak_from_this(), &Mine::on_input);
    }

    void on_input(int i) {
        std::cout << "Mine got i=" << i << '\n';
        if (i == 5) {
            gmine = nullptr;
        }
    }

    ~Mine() {
        std::cout << "Mine destruct\n";
    }
};

void dummy(Input &input) {
    auto bar = std::make_shared<Bar>();
    input.on_input.connect(std::weak_ptr<Bar>(bar), &Bar::on_input, pengsig::nshot_t(10)); // 调用 10 次后回调自动无效化
    gbar = bar;
    auto mine = std::make_shared<Mine>();
    mine->register_on(input);
    gmine = mine;
}

int main() {
    Input input;
    dummy(input);
    input.on_input.connect([=](int i) {
        std::cout << "main received input: " << i << '\n';
    });
    input.main_loop();
    return 0;
}
