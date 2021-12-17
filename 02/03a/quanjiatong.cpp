struct C {
    C();                       // 默认构造函数

    C(C const &c);             // 拷贝构造函数
    C(C &&c);                  // 移动构造函数（C++11 引入）
    C &operator=(C const &c);  // 拷贝赋值函数
    C &operator=(C &&c);       // 移动赋值函数（C++11 引入）

    ~C();                      // 解构函数
};
