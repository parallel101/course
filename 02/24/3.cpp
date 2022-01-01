#include <memory>

struct C {
    std::unique_ptr<C> m_child;
    C *m_parent;
};

int main() {
    auto parent = std::make_unique<C>();
    auto child = std::make_unique<C>();

    // 建立相互引用：
    child->m_parent = parent.get();
    parent->m_child = std::move(child);  // 移交 child 的所属权给 parent

    parent = nullptr;  // parent 会被释放。因为 child 指向他的是原始指针
    // 此时 child 也已经被释放了，因为 child 完全隶属于 parent

    return 0;
}
