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

void myfunc() {
    std::thread t1([&] {
        download("hello.zip"); 
    });
    t1.detach();
    // t1 所代表的线程被分离了，不再随 t1 对象销毁
}

int main() {
    myfunc();
    interact();
    return 0;
}
