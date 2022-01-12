#include <iostream>
#include <tbb/parallel_invoke.h>
#include <string>

void download(std::string file) {
    for (int i = 0; i < 10; i++) {
        std::cout << "Downloading " << file
                  << " (" << i * 10 << "%)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

int main() {
    tbb::parallel_invoke([&] {
        download("hello.zip");
    }, [&] {
        interact();
    });
    return 0;
}
