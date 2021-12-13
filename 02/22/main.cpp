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

    func(p);  // shared_ptr 允许拷贝！和当前指针共享所有权，引用计数加1

    printf("use count = %ld\n", p.use_count());   // 2

    func(p);  // 多次也没问题~ 多个 shared_ptr 会共享所有权，引用计数加1

    printf("use count = %ld\n", p.use_count());   // 3

    p->do_something();  // 正常执行，p 指向的地址本来就没有改变

    objlist.clear();    // 刚刚 p 移交给 func 的生命周期结束了！引用计数减2

    printf("use count = %ld\n", p.use_count());   // 1

    p->do_something();  // 正常执行，因为引用计数还剩1，不会被释放

    return 0;  // 到这里最后一个引用 p 也被释放，p 指向的对象才终于释放
}
