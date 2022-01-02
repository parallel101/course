#include <iostream>
#include <unistd.h>
#include <string>
#include <thread>

void download(std::string file) {
    for (int i = 0; i < 10; i++) {
        std::cout << "Downloading " << file
                  << " (" << i * 10 << "%)..." << std::endl;
        usleep(400000);
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

int main() {
    std::thread t1([&] {
        download("hello.zip"); 
    });
    interact();
    std::cout << "Waiting for child thread..." << std::endl;
    t1.join();
    std::cout << "Child thread exited!" << std::endl;
    return 0;
}
