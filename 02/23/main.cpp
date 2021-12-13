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

std::vector<std::shared_ptr<C>> objlist;

void func(std::shared_ptr<C> p) {
    objlist.push_back(std::move(p));  // 这里用移动可以更高效，但不必须
}

int main() {
    std::shared_ptr<C> p = std::make_shared<C>(); // 引用计数初始化为1

    printf("use count = %ld\n", p.use_count());   // 1

    std::weak_ptr<C> weak_p = p;        // 创建一个不影响计数器的弱引用

    printf("use count = %ld\n", p.use_count());   // 1

    func(std::move(p));  // 控制权转移，p 变为 null，引用计数加不变

    if (weak_p.expired())
        printf("错误：弱引用已失效！");
    else
        weak_p.lock()->do_something();  // 正常执行，p 的生命周期仍被 objlist 延续着

    objlist.clear();    // 刚刚 p 移交给 func 的生命周期结束了！引用计数减1，变成0了

    if (weak_p.expired())              // 因为 shared_ptr 指向的对象已释放，弱引用会失效
        printf("错误：弱引用已失效！");
    else
        weak_p.lock()->do_something();  // 不会执行

    return 0;  // 到这里最后一个弱引用 weak_p 也被释放，他指向的“管理块”被释放
}
