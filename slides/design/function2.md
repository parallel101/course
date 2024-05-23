# 回调函数·破局·设计模式

设计模式大都是在**面向对象**时代提出的，诚然，有一部分模式，天生适合面向对象的写法。

但是其中有一些（如下）写起来非常复杂而难懂，实际上是因为面向对象范式的固有缺陷。

- 命令模式
- 策略模式
- 观察者模式
- 访问者模式
- 装饰器模式
- 工厂模式
- 过滤器模式
- 构建者模式
- 代理模式

后来出现的**函数式编程范式**大大简化了这些。他创新性地引入了大名鼎鼎的**回调函数** (Delegate) 与**闭包** (Closure)。

古代 Java 是因为不支持函数式编程范式，只支持面向对象范式，才不得不用原始的繁琐的面向对象写法。

可恶的是，很多古板的所谓“设计模式”教程，有的甚至打着“现代 C++ 设计模式”的名号，还依然在以落后的**面向对象写法**来讲授这些明明更适合**函数式写法**的设计模式！

为此，小彭老师专门为你制作本期课程，以简化这些本就应该简单的设计模式。

我们会把函数式的写法和面向对象的写法都讲一遍，保证让你感受到函数式编程的敏捷魅力。

# 回调函数？

设想你产生了这样的需求：提交许多不同的命令到一个队列，然后一次性执行掉他们，这被称为**命令模式**。

> OpenGL 和 CUDA 都运用了命令模式。所有 gl 函数只是提交命令进入命令队列，等 glFlush 时才一次性提交到驱动里真正执行，节省了反复与驱动通信的开销。

```cpp
struct Command {
    virtual void run() = 0;
};

struct HelloCommand : Command {
    void run() override {
        puts("hello");
    }
};

struct WorldCommand : Command {
    void run() override {
        puts("world");
    }
};

vector<Command> commands;

commands.push_back(HelloCommand());
commands.push_back(WorldCommand());

for (auto &&command : commands) {
    command.run();
}
```

但上面的代码有个错误！涉及虚函数的类，必须总是以指针形态出现！

不能用 `vector<Command>`，必须用 `vector<Command *>`。否则 push_back 时会出现 object-slicing 问题，而且编译器会报错：

> error: cannot declare variable `commands` to be of abstract type `std::vector<Command>`

为了避免这个问题，你不得不用指针存储所有的命令对象：

```cpp
vector<Command *> commands;
```

这又会带来另一个麻烦：任务执行完毕后，你必须手动遍历 delete 掉所有的 commands！否则会有内存泄漏。

```cpp
commands.push_back(new HelloCommand());
commands.push_back(new WorldCommand());

for (auto &&command : commands) {
    command.run();
}

for (auto &&command : commands) {
    delete command;
}
```

为了避免手动 delete 的麻烦，你还得用智能指针：

```cpp
vector<shared_ptr<Command>> commands;

commands.push_back(make_shared<HelloCommand>());
commands.push_back(make_shared<WorldCommand>());

for (auto &&command : commands) {
    command->run();
}

commands.clear();  // clear 清空数组，就可以自动调用所有成员的析构函数，对于智能指针来说，就是自动调用 delete 了。
```

这样写，总算是没有问题了。但是，你有没有觉得，为了一个简单的命令，搞这么多仪式感，有点多余了？

回调函数就是为了解决虚函数的繁琐的而产生的。你只需要定义一个普通函数，无需大费周章定义接口并继承。

## 函数指针

在我们的例子里，我们 HelloCommand 和 WorldCommand 中真正执行的函数提取出来，成为全局函数。

函数类型一般会先定义一个别名，方便使用：

```cpp
typedef void Callback();  // 古代 C 语言
using Callback = void();  // 现代 C++
```

**函数本身并不能存储或作为参数传递，但是函数指针可以。**

因此 vector 里需要存储的应该是函数的指针 `Callback *`，而不是函数本身 `Callback`。

优势：和虚函数不一样的是，函数指针并不需要 delete，函数指针指向的是 `.text` 段，属于程序全局生命周期的指针，直到程序退出前都永不失效。

```cpp
void hello_func() {
    puts("hello");
}

void world_func() {
    puts("world");
}

using Callback = void();

vector<Callback *> commands;

commands.push_back(hello_func);
commands.push_back(world_func);

for (auto &&command : commands) {
    command();  // 不需要 ->run() 了，函数指针本身就可以调用
}
```

但是，C 语言的函数指针有一个缺点，他不能保存有任何状态变量！

```cpp
void number_func() {
    // 错误：访问不到 main 里的 i 变量
    printf("我保存的数字是 %d\n", i);
}

using Callback = void();

vector<Callback *> commands;

int main() {
    int i = 42;
    commands.push_back(number_func);

    for (auto &&command : commands) {
        command();
    }
}
```

## 仿函数

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void run() {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

number_func.run();
```

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void    operator括号     () {
        printf("我保存的数字是 %d\n", i);
    }
};

NumberFunc number_func(1);

number_func.  operator括号    ();
```

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void    operator()     () {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

number_func.  operator()    ();
```

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void    operator()     () {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

number_func();
```

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void operator()() {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

number_func();
```

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void operator()() const {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

number_func();
```

就这样 number_func 明明是个结构体，并不是一个函数，但是由于他重载了括号运算符，可以通过 `number_func()` 调用到 `number_func. operator() ()`。

这种不是函数，却能像函数一样用括号调用的结构体类型，被成为仿函数。

因为是个结构体，所以可以保存变量。因为有括号运算符重载，所以可以当函数一样使用。

然后，我们把过时的 C 语言函数指针 `void()` 改成更先进的 `std::function<void()>` 容器（也不用指针了！）

```cpp
// using Callback = void();  // 过时！

using Callback = function<void()>;

vector<Callback> commands;
```

> 注意这里不再需要是指针了，就和智能指针一样，function 会自动帮你管理所有内存！现代 C++ 不欢迎指针 :(

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void operator()() const {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);

commands.push_back(number_func);

for (auto &&command : commands) {
    command();
}
```

`function<void()>` 能够接受任何支持以 `f()` 形式调用的函数或仿函数。

```cpp
void myfunc() {
    puts("myfunc");
}

struct MyClass {
    void operator()() {
        puts("MyClass");
    }
};

function<void()> f1 = myfunc;
function<void()> f2 = MyClass();
```

`function<int(int)>` 能够接受任何支持以 `f(0)` 形式调用且返回 int 的函数或仿函数。

```cpp
int myfunc(int i) {
    return i + 1;
}

struct MyClass {
    int operator()(int i) {
        return i + 1;
    }
};

function<int(int)> f1 = myfunc;
function<int(int)> f2 = MyClass();
```

以此类推。

## Lambda 表达式

但是，仿函数的缺点是，有些时候你只需要定义一个临时的函数，用完就扔，不想定义一个结构体，这种时候要写一堆 `operator()()` 什么的就很麻烦。

于是，C++11 引入了 lambda 表达式，让你可以直接定义一个匿名仿函数。

```cpp
// lambda 表达式背后的仿函数结构体类型没有具体名字，只能通过 auto 捕获
auto number_func = [i = 42](int i) {
    printf("我保存的数字是 %d\n", i);
};
```

他完全等价于：

```cpp
struct NumberFunc {
    int i;

    NumberFunc(int i) : i(i) {}

    void operator()() const {
        printf("我保存的数字是 %d\n", i);
    }
};

auto number_func = NumberFunc(42);
```

这种捕获了状态的仿函数对象，被函数式思想家们称为闭包。

---

由于 lambda 和手动定义一个仿函数结构体无异。所以 lambda 类型都可以隐式转换到函数参数和返回值类型匹配的 function 容器里。

因此，你可以把 lambda 表达式产生的仿函数，直接添加到 commands 数组里。

lambda 要捕获的变量写在方括号里，如果没有要捕获的，那就留空。

```cpp
commands.push_back([]() {  // 没有捕获任何变量的 lambda 仿函数被称为无状态 lambda
    printf("我没有保存任何变量\n");
});
```

```cpp
commands.push_back([i = 42]() {
    printf("我保存的数字是 %d\n", i);
});
```

若要在 lambda 产生的仿函数结构体内保存多个成员变量，中间用逗号分割。

```cpp
commands.push_back([i = 42, j = 43]() {
    printf("我保存的数字分别是 %d 和 %d\n", i, j);
});
```

默认情况下，生成的仿函数的 `operator()` 函数都是 const 修饰的。

例如下面两段代码会编译出错。

```cpp
[i = 0] () {
    i = i + 1;  // 编译错误：无法在 const 成员函数里修改成员
    printf("我被调用了 %d 次\n", i);
};
```

等价于：

```cpp
struct NumberFunc {
    int i = 0;

    void operator()() const {
        i = i + 1;  // 编译错误：无法在 const 成员函数里修改成员
        printf("我被调用了 %d 次\n", i);
    }
};
```

要去掉生成的仿函数的 `operator()`，可以在 lambda 的参数列表后面加个 mutable。

```cpp
[i = 0] () mutable {
    i = i + 1;  // 编译通过，mutable 的 lambda 允许修改保存的变量值
    printf("我被调用了 %d 次\n", i);
};
```

等价于：

```cpp
struct NumberFunc {
    int i = 0;

    void operator()() {
        i = i + 1;  // 编译通过，非 const 的成员函数允许修改成员变量的值
        printf("我被调用了 %d 次\n", i);
    }
};
```

---

经常用到保存一个堆栈上变量的情况：

```cpp
int i = 42;
function<void(int)> add42 = [i = i] (int j) {
    return i + j;
};
```

每个变量都要 `i = i` 很麻烦，因此 lambda 允许你简写：

```cpp
int i = 42;
function<void(int)> add42 = [i] (int j) {
    return i + j;
};
```

当你需要在创建 lambda 时对做一个小修改：

```cpp
int i = 42;
function<void(int)> add42 = [i = i + 1] (int j) {  // 实际捕获到的变成了 42
    return i + j;
};
```

永远记住等号左侧是 lambda 内部捕获后的变量名，等号右侧可以是 main 函数栈上的变量，也可以是个任意表达式。

---

除了 mutable 以外，用引用保存栈上变量来修改，也是一个选择。

因此 lambda 的捕获表达式提供了 `&` 语法。

```cpp
int main() {
    int i = 0;
    function<void(int)> counter = [&i = i] () {
        printf("我被调用了 %d 次\n", i);  // OK: i 不是保存在 lambda 内部的变量，而是 main 函数栈上的变量
    };
    counter();
}
```

`[&i = i]` 这里等号前后变量名一致，也可以简写为 `[&i]`。

```cpp
int main() {
    int i = 0;
    function<void(int)> counter = [&i] () {
        i = i + 1;  // 编译通过：i 不是保存在 lambda 内部的变量，而是 main 函数栈上的变量，可以修改
        printf("我被调用了 %d 次\n", i);
    };
    counter();
}
```

等价于：

```cpp
struct Counter {
    int &i;  // 以引用方式保存

    Counter(int &i) : i(i) {}

    void operator()() const {
        i = i + 1;  // 编译通过：i 不是保存在结构体内部的变量，而是个指向外面引用，该引用不是 const 的，可以修改
        printf("我被调用了 %d 次\n", i);
    }
};

int main() {
    int i = 0;
    function<void(int)> counter = Counter(i); // 这里 i 传递的是引用哦
    counter();
}
```

---

可以既保存引用变量，又保存普通变量。

```cpp
int main() {
    int i = 0;
    int j = 0;
    function<void(int)> counter = [&i, j] () mutable {
        i = i + 1;
        j = j + 1;
    };
    counter();
    counter();
    counter();
    printf("i = %d, j = %d", i, j);  // i = 3, j = 0
}
```

可以看到这里通过引用保存的 i 对外面的 i 造成了影响。

## 函数化的策略模式

我们来改造昨天的 reduce 案例。

```cpp
int reduce(vector<int> v, Reducer *reducer) {
    int res = reducer.init();
    for (int i = 0; i < v.size(); i++) {
        res = reducer.add(res, v[i]);
    }
    return res;
}
```

需要用到一个接受两个 int 做参数，一个 int 做返回值的求和函数。

```cpp
int sum_add(int x, int y) {
    return x + y;
}
```

函数类型是 `int(int, int)`。

那么相应的 C++ 版的函数容器类型就是 `function< int(int, int) >`。

```cpp
int reduce(vector<int> v, function<int(int, int)> add, int init) {
    int res = init;
    for (int i = 0; i < v.size(); i++) {
        res = add(res, v[i]);
    }
    return res;
}

reduce(v, sum_add, 0);
```

但是 function 和虚函数一样，由于他是类型擦除容器，内部的实现一样是基于函数指针实现的动态多态，如果要运用在高性能并行编程中会成为性能瓶颈。

那刚才就属于是为了抽象牺牲了性能，如何不牺牲性能就实现抽象？

刚才说了，lambda 产生的仿函数类型，没有具体的类名，只能通过 auto 保存下来（auto 自动推导为右侧 lambda 的类型，不用写出类型的名字）。

```cpp
auto add = [] (int x, int y) {
    return x + y;
};
```

如果要写出具体名字，就只能是通用的函数容器 function 了：

```cpp
function<int(int, int)> add = [] (int x, int y) {
    return x + y;
};
```

既然**局部变量**可以用 **auto** 来自动推导类型。

同样地**函数参数**也可以利用**模板函数**推导类型。

```cpp
template <typename AddFunc>  // AddFunc 会自动推导为传入的 add 参数的类型
int reduce(vector<int> v, AddFunc add, int init) {
    int res = init;
    for (int i = 0; i < v.size(); i++) {
        res = add(res, v[i]);
    }
    return res;
}

auto add = [] (int x, int y) {  // lambda 表达式产生的匿名类型，没有名字，只能通过 auto 捕获
    return x + y;
};

reduce(v, add, 0);
// 等价于：
reduce<decltype(add)>(v, add, 0);
```

在 C++20 中，在函数参数中可以写出 auto，与定义一个模板函数参数等价：

```cpp
int reduce(vector<int> v, auto add, int init);
// 等价于：
template <typename AddFunc>
int reduce(vector<int> v, AddFunc add, int init);
```

就和局部变量的 auto 用起来一样，非常方便。
