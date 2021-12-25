#include <iostream>
#include <vector>

template <class Func>
void fetch_data(Func const &func) {
    for (int i = 0; i < 32; i++) {
        func(i);
        func(i + 0.5f);
    }
}

int main() {
    std::vector<int> res_i;
    std::vector<float> res_f;
    fetch_data([&] (auto const &x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, int>) {
            res_i.push_back(x);
        } else if constexpr (std::is_same_v<T, float>) {
            res_f.push_back(x);
        }
    });
    std::cout << res_i.size() << std::endl;
    std::cout << res_f.size() << std::endl;
    return 0;
}
