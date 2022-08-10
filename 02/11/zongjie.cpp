C c1 = c2;               // 拷贝构造函数
C c1 = std::move(c2);    // 移动构造函数

c1 = c2;                 // 拷贝赋值函数
c1 = std::move(c2);      // 移动赋值函数

C c1 = C();              // 移动构造函数
c1 = C();                // 移动赋值函数
return c2;               // 移动赋值函数
