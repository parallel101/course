#include <iostream>
#include <thread>
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
