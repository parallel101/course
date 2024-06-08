#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>

using namespace std;

template <class Callback>
struct Finally {
    Callback func;
    bool valid;

    Finally() : func(), valid(false) {}

    Finally(Callback func) : func(func), valid(true) {
    }

    Finally(Finally &&that) noexcept : func(std::move(that.func)), valid(that.valid) {
        that.valid = false; // 如果要支持移动语义，必须有个 bool 变量表示空状态！
    }

    Finally &operator=(Finally &&that) noexcept {
        if (this != &that) {
            if (valid) {
                func();
            }
            func = std::move(that.func);
            valid = that.valid;
            that.valid = false;
        }
        return *this;
    }

    void cancel() {
        valid = false;
    }

    void trigger() {
        if (valid) {
            func();
        }
        valid = false;
    }

    ~Finally() {
        if (valid) {
            func();
        }
    }
};

template <class Callback> // C++17 CTAD
Finally(Callback) -> Finally<Callback>;

int main() {
    Finally cb = [] {
        puts("调用了 Finally 回调");
    };
    srand(time(NULL));
    bool success = rand() % 2 == 0;
    if (success) {
        puts("操作失败，提前返回");
        // 此处提前返回导致析构，会自动 trigger
        return -1;
    }
    puts("操作成功");
    cb.cancel(); // cancel 后，析构不再自动 trigger 了
    return 0;
}
