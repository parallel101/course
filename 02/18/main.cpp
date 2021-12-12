#include <cstdio>
#include <memory>
#include <vector>

struct C {
    int m_number;

    C() {
        printf("分配内存!\n");
        m_number = 42;
    }

    ~C() {
        printf("释放内存!\n");
        m_number = -2333333;
    }

    void do_something() {
        printf("我的数字是 %d!\n", m_number);
    }
};

std::vector<std::unique_ptr<C>> objlist;

void func(std::unique_ptr<C> p) {
    objlist.push_back(std::move(p));  // 进一步移动到 objlist
}

int main() {
    std::unique_ptr<C> p = std::make_unique<C>();
    func(std::move(p));

    p->do_something();  // 出错，p 已经为空了！

    return 0;
}
