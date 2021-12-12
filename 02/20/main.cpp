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

    C *raw_p = p.get();
    func(std::move(p));

    raw_p->do_something();  // 正常执行，raw_p 保留了转移前的指针

    objlist.clear();        // 刚刚 p 移交给 func 的生命周期结束了！

    raw_p->do_something();  // 错误！raw_p 指向的对象已经被释放！

    return 0;
}
