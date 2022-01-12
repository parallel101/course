#include <iostream>
#include <tbb/parallel_invoke.h>
#include <string>

int main() {
    std::string s = "Hello, world!";
    char ch = 'd';
    tbb::parallel_invoke([&] {
        for (size_t i = 0; i < s.size() / 2; i++) {
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    }, [&] {
        for (size_t i = s.size() / 2; i < s.size(); i++) {
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    });
    return 0;
}
