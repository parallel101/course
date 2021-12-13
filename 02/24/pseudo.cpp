struct C {
    // 当一个类具有 unique_ptr 作为成员变量时：
    std::unique_ptr<D> m_pD;

    // 拷贝构造/赋值函数会被隐式地删除：
    // C(C const &) = delete;
    // C &operator=(C const &) = delete;

    // 移动构造/赋值函数不受影响：
    // C(C &&) = default;
    // C &operator=(C &&) = default;
};
