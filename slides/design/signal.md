# 函数指针

首先需要定义函数指针类型：

```cpp
using func_t = int (*)(int i);
// 等价于
typedef int (*func_t)(int i);
```

函数指针可以用于实现策略模式：

```cpp
void test(func_t func) {
    cout << func(2);
    cout << func(3);
}

int twice(int i) {
    return i * 2;
}

int main() {
    test(twice); // 4 6
}
```

## 函数指针实现多态

```cpp
int twice(int i) {
    return i * 2;
}

int triple(int i) {
    return i * 3;
}

int main() {
    test(twice);  // 4 6
    test(triple); // 6 9
}
```

## 函数指针的缺陷

缺点：函数指针只能指向全局函数，无法保存状态。

```cpp
int ntimes(int scale, int i) {
    return i * scale;
}

int main() {
    test(ntimes(2)); // 错误：没有这种语法
    test(ntimes(3)); // 错误：没有这种语法
}
```

# 函数调用运算符

```cpp
struct myclass {
    void run(string name) {
        cout << name << "，你好\n";
    }
};

myclass mc;
mc.run("小切");
```

```cpp
struct myclass {
    void operator() (string name) {
        cout << name << "，你好\n";
    }
};

myclass mc;
mc. operator() ("小切");
```

```cpp
struct myclass {
    void operator() (string name) {
        cout << name << "，你好\n";
    }
};

myclass mc;
mc("小切");
```

重载了 `operator()` 后的 myclass 对象，调用起来就和普通函数一样。

## 仿函数

仿函数是一种重载了**函数调用运算符**的类，可以像函数一样使用：

```cpp
struct twice_t {
    int operator() (int i) {
        return i * 2;
    }
};

void test(twice_t func) {
    cout << func(2);
    cout << func(3);
}

twice_t twice;

int main() {
    test(twice); // 4 6
}
```

仿函数的优势是可以保存**状态**：

```cpp
struct ntimes_t {
    int scale;

    ntimes_t(int scale) : scale(scale) {}

    int operator() (int i) {
        return i * scale;
    }
};

void test(ntimes_t func) {
    cout << func(2);
    cout << func(3);
}

int main() {
    ntimes_t twice(2);
    ntimes_t triple(3);
    ntimes_t quadric(4);
    test(twice);   // 4 6
    test(triple);  // 6 9
    test(quadric); // 8 12
}
```

## 仿函数的缺陷

缺点：需要在 test 中写明仿函数的具体类型，无法实现多态。

```cpp
struct twice_t {
    int operator() (int i) {
        return i * 2;
    }
};

struct triple_t {
    int operator() (int i) {
        return i * 2;
    }
};

void test(twice_t func) {
    cout << func(2);
    cout << func(3);
}

twice_t twice;
triple_t triple;

int main() {
    test(twice);  // 4 6
    test(triple); // 错误：test 只兼容了 twice_t 做参数
}
```

如何解决？

## 利用模板

```cpp
template <class Func>
void test(Func func) {
    cout << func(2);
    cout << func(3);
}

twice_t twice;
triple_t triple;

int main() {
    test(twice);  // 4 6
    test(triple); // 6 9
}
```

这样就实现了**编译期多态**，因为 test 函数的参数类型可以根据传入的具体函数类型进行推导。

## 模板传仿函数的缺陷

缺点：必须编译期确定，无法动态决定类型。

```cpp
template <class Func>
void test(Func func) {
    cout << func(2);
    cout << func(3);
}

twice_t twice;
triple_t triple;

int main() {
    bool ok;
    cin >> ok;
    test(ok ? twice : triple); // 错误：不兼容的类型之间不能三目
}
```

## 万能的 function 容器

std::function 采用了**类型擦除技术**，无需写明仿函数类的具体类型，能容纳任何仿函数或函数指针。

只需在模板参数中写明函数的参数和返回值类型即可，所有具有同样参数和返回值类型的仿函数或函数指针都可以传入。

```cpp
struct twice_t {
    int operator() (int i) {
        return i * 2;
    }
};

function<int (int)> twice = twice_t();  // 没问题，能接受仿函数


int triple(int i) {
    return i * 3;
}

function<int (int)> function_triple = triple; // 没问题，能接受函数指针


struct ntimes_t {
    int scale;

    ntimes_t(int scale) : scale(scale) {}

    int operator() (int i) {
        return i * scale;
    }
};

function<int (int)> quadric = ntimes_t(4); // 没问题，能接受带状态的仿函数
```

可以用 function 容器作为参数，就可以避免使用模板。

```cpp
// 模板：性能优先
template <class Func>
void test(Func func) {
    cout << func(2);
    cout << func(3);
}

// 容器：灵活性优先
void test(function<int (int)> func) {
    cout << func(2);
    cout << func(3);
}
```

## 函数式为什么好？

有人说，function 底层依然是基于函数指针实现的，不是和虚函数一样低效吗？函数式相比传统面向对象好在哪里呢？

1. 性能与灵活性的选择权

现实工程中，往往是 20% 的代码耗费了 80% 的计算机时间。我们只要优化这 20% 的瓶颈代码就可以。

虚函数实现的多态，是强制的，一旦用了虚函数，就没法把虚表去掉了，永远卡在你的对象类型里占 8 字节空间，永远只能以指针形态使用。

如果选择函数式编程范式，你可以在次要的业务逻辑代码中选择更灵活的 function 容器。

而在需要性能的瓶颈代码处，可以随时切换到基于模板的，更高性能的编译期多态。函数式给了你根据情况选择的自由度。

2. function 有小对象优化

而且 function 内部具有类似于 string 小对象优化机制，对于较小的状态或没有状态的仿函数，就无需指针！无需堆内存分配！

而虚函数哪怕没有状态，由于虚表指针的存在，也总是需要用 new 创建，总是会造成大量的碎片化内存，因此即使同样选择了灵活性，function 依然比虚函数高效一点。

而且即使你的状态非常多，导致 function 不得不需要堆内存分配了，这一分配也不用你自己操心，不用手动 new，也不用手动 make_shared，function 内部自动帮你完成一切。

3. lambda 表达式很方便

最重要的是，函数式编程范式可以便捷地利用 lambda 表达式就地创建仿函数对象，而面向对象需要大费周章定义一个类接口，然后再定义一个类实现虚函数，有时还需要分离声明和定义。

绕了个大圈子，不仅写起来痛苦，需要起名强迫症，而且看得人也头疼。lambda 表达式，就地创建，无需名字，更适合敏捷开发。
