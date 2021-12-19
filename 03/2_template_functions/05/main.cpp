#include <iostream>

template <int N>
void show_times(std::string msg) {
    for (int i = 0; i < N; i++) {
        std::cout << msg << std::endl;
    }
}

int main() {
    show_times<1>("one");
    show_times<3>("three");
    show_times<4>("four");
}
