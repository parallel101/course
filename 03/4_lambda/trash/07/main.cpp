#include <iostream>

struct print_t {
    template <class T>
    void operator()(T const &t) const {
        std::cout << t << std::endl;
    }
};
print_t print;

template <class Func>
void call_twice(Func const &func) {
    func(0);
    func(1);
    func(3.14f);
    func("Hello");
}

int main() {
    call_twice(print);
    return 0;
}
