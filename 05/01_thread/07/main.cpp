#include <iostream>
#include <thread>
#include <string>
#include <vector>

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

class ThreadPool {
    std::vector<std::thread> m_pool;

public:
    void push_back(std::thread thr) {
        m_pool.push_back(std::move(thr));
    }

    ~ThreadPool() {                      // main 函数退出后会自动调用
        for (auto &t: m_pool) t.join();  // 等待池里的线程全部执行完毕
    }
};

ThreadPool tpool;

void myfunc() {
    std::thread t1([&] {
        download("hello.zip");
    });
    // 移交控制权到全局的 pool 列表，以延长 t1 的生命周期
    tpool.push_back(std::move(t1));
}

int main() {
    myfunc();
    interact();
    return 0;
}
