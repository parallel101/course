---
theme: seriph
background: images/bg.png
class: text-center
highlighter: shiki
lineNumbers: true
info: |
  ## Archibate's STL course series
  Presentation slides for archibate.

  Learn more at [parallel101/course](https://github.com/parallel101/course)
drawings:
  persist: false
transition: none
css: unocss
---

# 小彭老师 深入浅出 STL 课程系列 之 map

让高性能数据结构惠及每一人

---
layout: two-cols
---

# 课程简介

😀😀😀

面向已经了解一定 C++ 语法，正在学习标准库的童鞋。

C++ 标准库又称 STL，包含了大量程序员常用的算法和数据结构，是 Bjarne Stroustrup 送给所有 C++ 程序员的一把瑞士军刀，然而发现很多童鞋并没有完全用好他，反而还被其复杂性误伤了。

如果你也对标准库一知半解，需要系统学习的话，那么本课程适合你。小彭老师将运用他特有的幽默答辩比喻，全面介绍各 STL 容器的所有用法。结合一系列实战案例，剖析常见坑点，使用技巧等。对比不同写法的性能与可读性，还会与 Python 语言相互类比方便记忆，科普的部分冷知识可以作为大厂面试加分项。

::right::

<center><img src="images/maijiaxiu.jpg" width="320" class="opacity-80"><p class="opacity-50">本课程受到童鞋一致好评</p></center>

---

# 课程亮点

👍👍👍

本系列课程与《侯杰老师 STL 课》的区别：

- 侯杰老师价值 2650 元，本课程录播上传 B 站免费观看，观众可以自行选择是否一键三连。
- 课件和案例源码开源，上传在 GitHub，可以自己下载来做修改，然后自己动手实验。
- 侯杰老师注重理论和底层实现原理，而本课程注重应用，结合实战案例，着重展开重难点。
- 很多学校里教的，百度上搜的，大多是老版本 C++，已经过时了，而本课程基于较新的 C++17 和 C++20 语言标准。
- 有时存在部分 C++ 高级用法过于艰深，不能适合所有同学，本课程采用因材施教思想：对于新手，可以跳过看不懂的部分，看我提供的“保底用法”，不保证高性能和“优雅”，但至少能用；对学有余力的童鞋，则可以搏一搏上限，把高级用法也看懂，提升面试竞争力。总之不论你是哪个阶段的学习者，都能从此课程中获益。

---

# 课程大纲

✨✨✨

之前几期课程的录播已经上传到比站了[^1]。

1. vector 容器初体验 & 迭代器入门 (BV1qF411T7sd)
2. 你所不知道的 set 容器 & 迭代器分类 (BV1m34y157wb)
3. string，string_view，const char * 的爱恨纠葛 (BV1ja411M7Di) 
4. 万能的 map 容器全家桶及其妙用举例 (本期)
5. 函子 functor 与 lambda 表达式知多少
6. 通过实战案例来学习 STL 算法库
7. C++ 标准输入输出流 & 字符串格式化
8. traits 技术，用户自定义迭代器与算法
9. allocator，内存管理与对象生命周期
10. C++ 异常处理机制的前世今生

[^1]: https://space.bilibili.com/263032155

---

# 课程要求

✅✅✅

课件中所示代码推荐实验环境如下：

| 要求     | ❤❤❤        | 💣💣💣       | 💩💩💩         |
|----------|------------|--------------|----------------|
| 操作系统 | Arch Linux | Ubuntu 20.04 | Wendous 10     |
| 知识储备 | 会一点 C++ | 大学 C 语言  | Java 面向对象  |
| 编译器   | GCC 9 以上 | Clang 12     | VS 2019        |
| 构建系统 | CMake 3.18 | 任意 C++ IDE | 命令行手动编译 |

❤ = 推荐，💣 = 可用，💩 = 困难

---

# 如何使用课件

🥰🥰🥰

本系列课件和源码均公布在：https://github.com/parallel101/course

例如本期的课件位于 `course/stlseries/stl_map/slides.md`。

课件基于 Slidev[^1] 开发，Markdown 格式书写，在浏览器中显示，在本地运行课件需要 Node.js：

- 运行命令 `npm install` 即可自动安装 Slidev
- 运行命令 `npm run dev` 即可运行 Slidev 服务
- 浏览器访问 http://localhost:3030 即可看到课件

如果报错找不到 `slidev` 命令，可以试试 `export PATH="$PWD/node_modules/.bin:$PATH"`。

如果不想自己配置 Node.js 也可以直接以文本文件格式打开 slides.md 浏览课件。

Slidev 服务运行时对 slides.md 的所有修改会立刻实时显现在浏览器中。

[^1]: https://sli.dev/

---

# 如何运行案例代码

🥺🥺🥺

CMake 工程位于课件同目录的 `course/stlseries/stl_map/experiment/` 文件夹下。

其中 main.cpp 仅导入运行案例所需的头文件，具体各个案例代码分布在 slides.md 里。

如需测试具体代码，可以把 slides.md 中的案例代码粘贴到 main.cpp 的 main 函数体中进行实验。

附赠的一些实用头文件，同鞋们可以下载来在自己的项目里随意运用。

| 文件名 | 功能 |
|-|-|
| main.cpp | 把 slides.md 中你想要实验的代码粘贴到 main 函数中 |
| print.h | 内含 print 函数，支持打印绝大多数 STL 容器，方便调试 |
| cppdemangle.h | 获得类型的名字，以模板参数传入，详见该文件中的注释 |
| map_get.h | 带默认值的 map 表项查询，稍后课程中会介绍到 |
| ppforeach.h | 基于宏的编译期 for 循环实现，类似于 BOOST_PP_FOREACH |

---

# 课程书写习惯说明

```cpp
int const &i  // 本课程书写习惯
const int& i  // 官方文档书写习惯
```

---

# 本期课程目录

---

为什么需要 map

---

标准库中的 map 容器[^1]

[^1]: https://en.cppreference.com/w/cpp/container/map

---

特性：

- 有序
- 不重复

---

```cpp
using namespace std;
```

---

创建一个 map 对象：

```cpp
map<string, int> config;
```

一开始 map 默认是空的，如何插入一些初始数据？

```cpp
config["timeout"] = 985;
config["delay"] = 211;
```

过于落后，还有更新更专业的写法。

---

C++11 新特性——花括号初始化列表，允许创建 map 时直接指定初始数据：

```cpp
map<string, int> config = { {"timeout", 985}, {"delay", 211} };
```

通常我们会换行写，一行一个键值对，看起来条理更清晰：

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
```

总结：

```cpp
map<K, V> m = {
    {k1, v1},
    {k2, v2},
    ...,
};
```

---

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
```

等号可以省略（这其实就是在调用 map 的构造函数）：

```cpp
map<string, int> config{
    {"timeout", 985},
    {"delay", 211},
};
```

也可以先构造再赋值给 auto 变量：

```cpp
auto config = map<string, int>{
    {"timeout", 985},
    {"delay", 211},
};
```

都是等价的。

> 关于构造函数、花括号列表的具体语法可以参考我的《高性能并行》系列第二课：https://www.bilibili.com/video/BV1LY411H7Gg

---

作为函数参数时，可以用花括号初始化列表就地构造一个 map 对象：

```cpp
void myfunc(map<string, int> config);  // 函数声明

myfunc(map<string, int>{               // 直接创建一个 map 传入
    {"timeout", 985},
    {"delay", 211},
});
```

由于 `myfunc` 函数具有唯一确定的重载，要构造的参数类型 `map<string, int>` 可以省略不写：

```cpp
myfunc({
    {"timeout", 985},
    {"delay", 211},
});
```

函数这边，通常还会加上 `const &` 修饰避免不必要的拷贝。

```cpp
void myfunc(map<string, int> const &config);
```

---

从 vector 中批量导入键值对：

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config(kvs.begin(), kvs.end());
```

与刚刚花括号初始化的写法等价，只不过是从现有的 vector 中导入。同样的写法也适用于从 array 导入。

> 如果记不住这个写法，也可以自己手写 for 循环遍历 vector 逐个逐个插入 map，效果是一样的。

冷知识，如果不是 vector 而是想从传统的 C 语言数组中导入：

```cpp
pair<string, int> kvs[] = {
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config(kvs, kvs + 2);                    // C++98
map<string, int> config(std::begin(kvs), std::end(kvs));  // C++17
```

> 其中 `std::begin` 和 `std::end` 为 C++17 新增函数，专门用于照顾没法有成员函数 `.begin()` 的 C 语言数组。类似的全局函数还有 `std::size` 和 `std::data` 等……他们都是既兼容 STL 容器也兼容 C 数组的。


---

如何根据键查询相应的值？

很多同学都知道 map 具有 [] 运算符重载，和一些脚本语言一样，直观易懂。

```cpp
config["timeout"] = 985;       // 把 config 中键 timeout 对应值设为 985
auto val = config["timeout"];  // 读取 config 中键 timeout 对应值
print(val);                    // 985
```

但其实用 [] 访问元素是很不安全的，下面我会做实验演示这一点。

---

沉默的 []，无言的危险：当键不存在时，会返回 0 而不会出错！

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config["timeout"]); // 985
print(config["tmeout"]);  // 默默返回 0
```

```
985
0
```

当查询的键值不存在时，[] 会默默创建并返回 0。

这非常危险，例如一个简简单单的拼写错误，就会导致 map 的查询默默返回 0，你还在那里找了半天摸不着头脑，根本没发现错误原来在 map 这里。

---

爱哭爱闹的 at()，反而更讨人喜欢

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config.at("timeout"));  // 985
print(config.at("tmeout"));   // 该键不存在！响亮地出错
```

```
985
terminate called after throwing an instance of 'std::out_of_range'
  what():  map::at
Aborted (core dumped)
```

有经验的老手都明白一个道理：**及时奔溃**比**容忍错误**更有利于调试。即 fail-early, fail-loudly[^1] 原则。

例如 JS 和 Lua 的 [] 访问越界不报错而是返回 undefined / nil，导致实际出错的位置在好几十行之后，无法定位到真正出错的位置，这就是为什么后来发明了错误检查更严格的 TS。

使用 at() 可以帮助你更容易定位到错误，是好事。

[^1]: https://oncodingstyle.blogspot.com/2008/10/fail-early-fail-loudly.html

---

[] 更危险的地方在于，当所查询的键值不存在时：

```cpp
map<string, int> config = {
    {"timeout", 985},
    {"delay", 211},
};
print(config);
print(config["tmeout"]);  // 有副作用！
print(config);
```

```
{"delay": 211, "timeout": 985}
0
{"delay": 211, "timeout": 985, "tmeout": 0}
```

会自动创建那个不存在的键值！

你以为你只是观察了一下 map 里的 "tmeout" 元素，却意外改变了 map 的内容，薛定谔[^1]直呼内行。

[^1]: https://baike.baidu.com/item/%E8%96%9B%E5%AE%9A%E8%B0%94%E7%9A%84%E7%8C%AB/554903

---

> 在官方文档和各种教学课件中，都会展示一个函数的“原型”来讲解。
>
> 原型展现了一个函数的名称，参数类型，返回类型等信息，掌握了函数的原型就等于掌握了函数的调用方法。
>
> 本课程后面也会大量使用，现在来教你如何看懂成员函数的原型。

假设要研究的类型为 `map<K, V>`，其中 K 和 V 是模板参数，可以替换成你具体的类型。

例如当我使用 `map<string, int>` 时，就把下面所有的 K 替换成 string，V 替换成 int。

`map<K, V>` 的 [] 和 at 员函数，原型如下：

```cpp
V &operator[](K const &k);
V &at(K const &k);                   // 第一个版本的 at
V const &at(K const &k) const;       // 第二个版本的 at
```

可见 operator[] 只有一个版本，at 居然有名字相同的两个！这样不会发生冲突吗？

这是利用了 C++ 的“重载”功能，重载就是同一个函数有多个不同的版本，各个版本的参数类型不同。

---

> 例如小彭老师打电话给 110，假如警察叔叔发现小彭老师报的案子是网络诈骗，那么他们会帮我转接到网警部门；假如发现小彭老师是被绑架了，那么他们可能会出动武警解救小彭老师。这就是 110 函数的两个重载，根据调用者传入的信息类型，决定要转给哪一个子部门。

同理，编译器也是会根据调用时你传入的参数类型，决定要调用重载的哪一个具体版本。

- C 语言没有重载，函数名字相同就会发生冲突，编译器会当场报错。
- C++ 支持重载，只有当函数名字相同，参数列表也相同时，才会发生冲突。
- 返回值类型不影响重载，重载只看参数列表。

菜鸟教程上对 C++ 重载的解释[^1]：

> C++ 允许在同一作用域中的某个函数和运算符指定多个定义，分别称为函数重载和运算符重载。
>
> 重载声明是指一个与之前已经在该作用域内声明过的函数或方法具有相同名称的声明，但是它们的参数列表和定义（实现）不相同。
>
> 当您调用一个重载函数或重载运算符时，编译器通过把您所使用的参数类型与定义中的参数类型进行比较，决定选用最合适的定义。选择最合适的重载函数或重载运算符的过程，称为重载决策。
>
> 在同一个作用域内，可以声明几个功能类似的同名函数，但是这些同名函数的形式参数（指参数的个数、类型或者顺序）必须不同。您不能仅通过返回类型的不同来重载函数。

[^1]: https://www.runoob.com/cplusplus/cpp-overloading.html

---

```cpp
V &at(K const &k);                   // 第一个版本的 at
V const &at(K const &k) const;       // 第二个版本的 at
```

但是上面这两个 at 函数的参数类型都是 `K const &`，为什么可以重载呢？

注意看第二个版本最后面多了一个 const 关键字，这种写法是什么意思？小彭老师对其进行祛魅化：

```cpp
V &at(map<K, V> *this, K const &k);                   // 第一个版本的 at
V const &at(map<K, V> const *this, K const &k);       // 第二个版本的 at
```

原来实际上在函数括号后面加的 const，实际上是用于修饰 this 指针的！

> 该写法仅供示意，并非真实语法

所以两个 at 的参数列表不同，不同在于传入 this 指针的类型，所以可以重载，不会冲突。

- 当 map 对象为 const 时，传入的 this 指针为 `map<K, V> const *`，所以只能调用第二个版本的 at。
- 当 map 对象不为 const 时，传入的 this 指针为 `map<K, V> *`，两个重载都可以调用，但由于第一个重载更加符合，所以会调用第一个版本的 at。

---

刚刚解释了函数重载，那么运算符重载呢？

因为原本 C 语言就有 [] 运算符，不过那只适用于原始指针和原始数组。而 C++ 允许也 [] 运算符支持其他用户自定义类型（比如 std::map），和 C 语言自带的相比就只有参数类型不同（一个是原始数组，一个是 std::map），所以和函数重载很相似，这就是运算符重载。

```cpp
m["key"];
```

会被编译器“翻译”成：

```cpp
m.operator[]("key");
```

以上代码并非仅供示意，是可以通过编译运行之一。你还可以试试 `string("hel").operator+("lo")`。

> operator[] 虽然看起来很复杂一个关键字加特殊符号，其实无非就是个特殊的函数名，学过 Python 的童鞋可以把他想象成 `__getitem__`

```cpp
V &operator[](K const &k);
```

结论：[] 运算符实际上是在调用 operator[] 函数。

---

而 operator[] 这个成员函数没有 const 修饰，因此当 map 修饰为 const 时编译会不通过[^1]：

```cpp
const map<string, int> config = {  // 此处如果是带 const & 修饰的函数参数也是同理
    {"timeout", 985},
    {"delay", 211},
};
print(config["timeout"]);          // 编译出错
```

```
/home/bate/Codes/course/stlseries/stl_map/experiment/main.cpp: In function ‘int main()’:
/home/bate/Codes/course/stlseries/stl_map/experiment/main.cpp:10:23: error: passing ‘const std::map<std::__cxx11::basic_string<char>, int>’ as ‘this’ argument discards qualifiers [-fpermissive]
   10 | print(config["timeout"]);
```

编译器说 discards qualifiers，意思是 map 有 const 修饰，但是 operator[] 没有。

这实际上就是在说：`map<K, V> const *` 不能转换成 `map<K, V> *`。

有 const 修饰的 map 作为 this 指针传入没 const 修饰的 operator[] 函数，是减少了修饰（discards qualifers）。C++ 规定传参时只能增加修饰不能减少修饰：只能从 `map *` 转换到 `map const *` 而不能反之，所以对着一个 const map 调用非 const 的成员函数 operator[] 就出错了。相比之下 at() 就可以在 const 修饰下编译通过。

[^1]: https://blog.csdn.net/benobug/article/details/104903314

---

既然 [] 这么危险，为什么还要存在呢？

```cpp
map<string, int> config = {
    {"delay", 211},
};
config.at("timeout") = 985;  // 键值不存在，报错！
config["timeout"] = 985;     // 成功创建并写入 985
```

因为当我们写入一个本不存在的键值的时候，恰恰需要他的“自动创建”这一特性。

总结：读取时应该用 at() 更安全，写入时才需要用 []。

---

- 读取元素时，统一用 at()
- 写入元素时，统一用 []

```cpp
auto val = m.at("key");
m["key"] = val;
```

为什么其他语言比如 Python，只有一个 [] 就行了呢？而 C++ 需要两个？

- 因为 Python 会检测 [] 位于等号左侧还是右侧，根据情况分别调用 `__getitem__` 或者 `__setitem__`。
- C++ 编译器没有这个特殊检测，也检测不了，因为 C++ 的 [] 只是返回了个引用，并不知道 [] 函数返回以后，你是拿这个引用写入还是读取。为了保险起见他默认你是写入，所以先帮你创建了元素，返回这个元素的引用，让你写入。
- 而 Python 的引用是不能用 = 覆盖原值的，那样只会让变量指向新的引用，只能用 .func() 引用成员函数或者 += 才能就地修改原变量，这是 Python 这类脚本语言和 C++ 最本质的不同。
- 总而言之，我们用 C++ 的 map 读取元素时，需要显式地用 at() 告诉编译器我是打算读取。

---

at 与 [] 实战演练

我们现在的甲方是一个学校的大老板，他希望让我们管理学生信息，因此需要建立一个映射表，能够快速通过学生名字查询到相应的学生信息。思来想去 C++ 标准库中的 map 容器最合适。决定设计如下：

- 键为学生的名字，string 类型。
- 值为一个自定义结构体，Student 类型，里面存放各种学生信息。

然后自定义一下 Student 结构体，现在把除了名字以外的学生信息都塞到这个结构体里。

创建 `map<string, Student>` 对象，变量名为 `stus`，这个 map 就是甲方要求的学生表，成功交差。

```cpp
struct Student {
    int id;             // 学号
    int age;            // 年龄
    string sex;         // 性别
    int money;          // 存款
    set<string> skills; // 技能
};

map<string, Student> stus;
```

---

现在小彭老师和他的童鞋们要进入这家学校了，让我们用 [] 大法插入他的个人信息：

```cpp
stus["彭于斌"] = Student{20220301, 22, "自定义", {"C", "C++"}};
stus["相依"] = Student{20220301, 21, "男", 2000, {"Java", "C"}};
stus["樱花粉蜜糖"] = Student{20220301, 20, "女", 3000, {"Python", "CUDA"}};
stus["Sputnik02"] = Student{20220301, 19, "男", 4000, {"C++"}};
```

由于 C++11 允许省略花括号前的类型不写，所以 Student 可以省略，简写成：

```cpp
stus["彭于斌"] = {20220301, 22, "自定义", {"C", "C++"}};
stus["相依"] = {20220301, 21, "男", 2000, {"Java", "C"}};
stus["樱花粉蜜糖"] = {20220301, 20, "女", 3000, {"Python", "CUDA"}};
stus["Sputnik02"] = {20220301, 19, "男", 4000, {"C++"}};
```

又由于 map 支持在初始化时就指定所有元素，我们直接写：

```cpp
map<string, Student> stus = {
    {"彭于斌", {20220301, 22, "自定义", 1000, {"C", "C++"}}},
    {"相依", {20220301, 21, "男", 2000, {"Java", "C"}}},
    {"樱花粉蜜糖", {20220301, 20, "女", 3000, {"Python", "CUDA"}}},
    {"Sputnik02", {20220301, 19, "男", 4000, {"C++"}}},
};
```

---

现在甲方要求添加一个“培训”函数，用于他们的 C++ 培训课。

培训函数的参数为字符串，表示要消费学生的名字。如果该名字学生不存在，则应该及时报错。

每次培训需要消费 2650 元，消费成功后，往技能 skills 集合中加入 "C++"。

```cpp
void PeiXunCpp(string stuName) {
    auto stu = stus.at(stuName);  // 在栈上拷贝了一份完整的 Student 对象
    stu.money -= 2650;
    stu.skills.insert("C++");
}
```

然而，这样写是不对的！

`stus.at(stuName)` 返回的是一个引用 `Student &`，但是等号左侧，却不是个引用，而是普通变量。

那么这时会调用 Student 的拷贝构造函数，`Student(Student const &)`，来初始化变量 stu。

结论：把引用保存到普通变量中，则引用会退化，造成深拷贝！stu 和 stus.at(stuName) 的已经是两个不同的 Student 对象，对 stu 的修改不会影响到 stus.at(stuName) 指向的那个 Student 对象了。


此时你对这个普通变量的所有修改，都不会同步到 map 中的那个 Student 中去！

---

我们现在对相依童鞋进行 C++ 培训：

```cpp
PeiXunCpp("相依");
print(stus.at("相依"));
```

结果发现他的存款一分没少，也没学会 C++：

```
{id: 20220302, age: 21, sex: "男", money: 2000, skills: {"C", "Java"}}
```

看来我们的修改没有在 map 中生效？原来是因为我们在 PeiXunCpp 函数里：

```cpp
auto stu = stus.at(stuName);  // 在栈上拷贝了一份完整的 Student 对象
```

一不小心就用了“克隆人”技术！从学生表里的“相依1号”，克隆了一份放到栈上的“相依2号”！

然后我们扣了这个临时克隆人“相依2号”的钱，并给他培训 C++ 技术。

然而我们培训的是栈上的临时变量“相依2号”，克隆前的“相依1号”并没有受到培训，也没有扣钱。

然后呢？残忍的事情发生了！在小彭老师一通操作培训完“相依2号”后，我们把他送上断头台——析构了！

而这一切“相依1号”完全不知情，他只知道有人喊他做克隆，然后就回家玩 Java 去了，并没有培训 C++ 的记忆。

---

要防止引用退化成普通变量，需要把变量类型也改成引用！这种是浅拷贝，stu 和 stus.at(stuName) 指向的仍然是同一个 Student 对象。用 `auto` 捕获的话，改成 `auto &` 就行。

```cpp
void PeiXunCpp(string stuName) {
    auto &stu = stus.at(stuName);  // 在栈上创建一个指向原 Student 对象的引用
    stu.money -= 2650;
    stu.skills.insert("C++");
}
```

```
{id: 20220302, age: 21, sex: "男", money: -650, skills: {"C", "C++", "Java"}}
```

终于，正版“相依1号”本体鞋废了 C++！

之后如果再从“相依1号”身上克隆，克隆出来的“相依n号”也都会具有培训过 C++ 的记忆了。

引用相当于身份证，我们复印了“相依”的身份证，身份证不仅复印起来比克隆一个大活人容易（拷贝开销）从而提升性能，而且通过身份证可以找到本人，对身份证的修改会被编译器自动改为对本人的修改，例如通过“相依”的身份证在银行开卡等，银行要的是身份证，不是克隆人哦。

---

引用是一个烫手的香香面包，普通变量就像一个臭臭的答辩马桶，把面包放到马桶（auto）里，面包就臭掉，腐烂掉，不能吃了！要让面包转移阵地了以后依然好吃，需要放到保鲜盒（auto &）里。

这就是 C++ 的 decay（中文刚好是“退化”、“变质”的意思）规则。

以下都是香香面包，放进马桶里会变质：

- `T &` 会变质成 `T`（引用变质成普通变量）
- `T []` 会变质成 `T *`（数组变质成首地址指针）
- `T ()` 会变质成 `T (*)()`（函数变质成函数指针）

在函数的参数中、函数的返回值中、auto 捕获的变量中，放入这些香香面包都会发生变质！

如何避免变质？那就不要用马桶（普通变量）装面包呗！用保鲜盒（引用）装！

- 避免引用 `T &t` 变质，就得把函数参数类型改成引用，或者用 `auto &`，`auto const &` 捕获才行。
- 避免原生数组 `T t[N]` 变质，也可以改成引用 `T (&t)[N]`，但比较繁琐，不如直接改用 C++11 封装的安全静态数组 `array<T, N>` 或 C++98 就有的安全动态数组 `vector<T>`。
- 避免函数 `T f()` 变质，可以 `T (&f)()`，但繁琐，不如直接改用 C++11 的函数对象 `function<T()>`。

---

邪恶的 decay 规则造成空悬指针的案例

```cpp
typedef double arr_t[10];

auto func(arr_t val) {
    arr_t ret;
    memcpy(ret, val, sizeof(arr_t));  // 对 val 做一些运算, 把计算结果保存到 ret
    return ret;     // double [10] 自动变质成 double *
}

int main() {
    arr_t val = {1, 2, 3, 4};
    auto ret = func(val);             // 此处 auto 会被推导为 double *
    print(std::span<double>(ret, ret + 10));
    return 0;
}
```

```
Segmentation fault (core dumped)
```

---

修复方法：不要用沙雕 C 语言的原生数组，用 C++ 封装好的 array

```cpp
typedef std::array<double, 10> arr_t;  // 或者 vector 亦可

auto func(arr_t val) {
    arr_t ret;
    ret = val;  // 对 val 做一些运算, 把计算结果保存到 ret
    return ret;
}

int main() {
    arr_t val = {1, 2, 3, 4};
    auto ret = func(val);
    print(ret);
    return 0;
}
```

```
{1, 2, 3, 4, 0, 0, 0, 0, 0, 0}
```

---

当然如果你还是学不会怎么保留香香引用的话，也可以在修改后再次用 [] 写回学生表。这样学生表里不会 C++ 的“相依1号”就会被我们栈上培训过 C++ 的“相依1号”覆盖，现在学生表里的也是有 C++ 技能的相依辣！只不过需要翻来覆去克隆了好几次比较低效而已，至少能用了，建议只有学不懂引用的童鞋再用这种保底写法。

```cpp
void PeiXunCpp(string stuName) {
    auto stu = stus.at(stuName);  // 克隆了一份“相依2号”
    stu.money -= 2650;
    stu.skills.insert("C++");
    stus[stuName] = stu;          // “相依2号”夺舍，把“相依1号”给覆盖掉了
}
```

学生思考题：上面代码第 5 行也可以改用 at，为什么？小彭老师不是说 “at 用于读取，[] 用于写入” 吗？

我们童鞋要学会变通！小彭老师说 [] 用于写入，是因为有时候我们经常需要写入一个不存在的元素，所以 [] 会自动创建元素而不是出错就很方便；但是现在的情况是我们第 2 行已经访问过 at("相依")，那么就确认过 "相依" 已经存在才对，因此我写入的一定是个已经存在的元素，这时 [] 和 at 已经没区别了，所以用 at 的非 const 那个重载，一样可以写入。

我们童鞋不是去死记硬背《小彭老师语录》，把小彭老师名言当做“两个凡是”圣经。要理解小彭老师会这么说的原因是什么，这样才能根据不同实际情况，实事求是看问题，才是符合小彭老师唯物编程观的。

---

如果要根据学号进行查找呢？那就以学号为键，然后把学生姓名放到 Student 结构体中。

如果同时有根据学号进行查找和根据姓名查找两种需求呢？

同时高效地根据多个键进行查找，甚至指定各种条件，比如查询所有会 C++ 的学生等，这可不是 map 能搞定的，或者说能搞定但不高效（只能暴力遍历查找，间复杂度太高）。这是个专门的研究领域，称为：关系数据库。

关系数据库的实现有 MySQL，SQLite，MongoDB 等。C++ 等编程语言只需调用他们提供的 API 即可，不必自己手动实现这些复杂的查找和插入算法。

这就是为什么专业的“学生管理系统”都会用关系数据库，而不是自己手动维护一个 map，因为关系数据库的数据结构更复杂，但经过高度封装，提供的功能也更全面，何况 map 在内存中，电脑一关机，学生数据就没了。

---

你知道吗？[] 的妙用

除了写入元素需要用 [] 以外，还有一些案例中合理运用 [] 会非常的方便。

[] 的效果：当所查询的键值不存在时，会调用默认构造函数创建一个元素[^1]。

- 对于 int, float 等数值类型而言，默认值是 0。
- 对于指针（包括智能指针）而言，默认值是 nullptr。
- 对于 string 而言，默认值是空字符串 ""。
- 对于 vector 而言，默认值是空数组 {}。
- 对于自定义类而言，会调用你写的默认构造函数，如果没有，则每个成员都取默认值。

[^1]: https://en.cppreference.com/w/cpp/language/value_initialization

---

[] 妙用举例：出现次数统计

```cpp
vector<string> input = {"hello", "world", "hello"};
map<string, int> counter;
for (auto const &key: input) {
    counter[key]++;
}
print(counter);
```

```
{"hello": 2, "world": 1}
```

---
layout: two-cols-header
---

::left::

<center>

活用 [] 自动创建 0 元素的特性

</center>

```cpp {3}
map<string, int> counter;
for (auto const &key: input) {
    counter[key]++;
}
```

::right::

<center>

古板的写法

</center>

```cpp {3-7}
map<string, int> counter;
for (auto const &key: input) {
    if (!counter.count(key)) {
        counter[key] = 1;
    } else {
        counter[key] = counter.at(key) + 1;
    }
}
```

---

[] 妙用举例：归类

```cpp
vector<string> input = {"happy", "world", "hello", "weak", "strong"};
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    categories[key].push_back(str);
}
print(categories);
```

```
{'h': {"happy", "hello"}, 'w': {"world", "weak"}, 's': {"strong"}}
```

---
layout: two-cols-header
---

::left::

<center>

活用 [] 自动创建"默认值"元素的特性

</center>

```cpp {4}
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    categories[key].push_back(str);
}
print(categories);
```

::right::

<center>

古板的写法

</center>

```cpp {4-8}
map<char, vector<string>> categories;
for (auto const &str: input) {
    char key = str[0];
    if (!categories.count(key)) {
        categories[key] = {str};
    } else {
        categories[key].push_back(str);
    }
}
```

---
layout: center
---

![Elegence](https://pica.zhimg.com/50/v2-f2560f634b1e09f81522f29f363827f7_720w.jpg)

---

查询 map 中元素的数量

---

应用举例：给每个键一个独一无二的计数

---

判断是否存在：count 函数

```cpp
size_t count(K const &k) const;
```

count 返回容器中键和参数 k 相等的元素个数，类型为 size_t（无符号 64 位整数）。

由于 map 中同一个键最多只可能有一个元素，取值只能为 0 或 1。

并且 size_t 可以隐式转换为 bool 类型，0 则 false，1 则 true。

因此可以直接通过 count 的返回值是否为 0 判断一个键在 map 中是否存在：

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
if (msg.count("fuck")) {
    print("存在fuck，其值为", msg.at("fuck"));
} else {
    print("找不到fuck");
}
if (msg.count("suck")) {
    print("存在suck，其值为", msg.at("suck"));
} else {
    print("找不到suck");
}
```

C++20 可以改用返回类型为 bool 的 contains 函数，满足你的所有强迫症。

---

STL 容器的元素类型都可以通过成员 `value_type` 查询，常用于泛型编程（又称元编程）。

```cpp
set<int>::value_type      // int
vector<int>::value_type   // int
string::value_type        // char
```

在本课程的案例代码中，附赠一份 "cppdemangle.h"，可以实现根据指定的类型查询类型名称并打印出来。

跨平台，支持 MSVC，Clang，GCC 三大编译器，例如：

```cpp
int i;
print(cppdemangle<decltype(std::move(i))>());
print(cppdemangle<std::string>());
print(cppdemangle<std::wstring::value_type>());
```

在我的 GCC 12.2.1 上得到：

```
"int &&"
"std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >"
"wchar_t"
```

---

问题: map 真正的元素类型究竟是什么？其具有三个成员类型[^1]：

- 元素类型：`value_type`
- 键类型：`key_type`
- 值类型：`mapped_type`

用 cppdemangle 做实验，看看这些成员类型具体是什么吧：

```cpp
map<int, float>::value_type   // pair<const int, float>
map<int, float>::key_type     // int
map<int, float>::mapped_type  // float
```

结论：`map<K, V>` 的元素类型是 `pair<const K, V>` 而不是 `V`。

[^1]: https://blog.csdn.net/janeqi1987/article/details/100049597

---

`pair<const K, V>` ——为什么 K 要加 const？

上期 set 课说过，set 内部采用红黑树数据结构保持有序，这样才能实现在 $O(\log N)$ 时间内高效查找。

键值改变的话会需要重新排序，如果只修改键值而不重新排序，会破坏有序性，导致二分查找结果错误！
所以 set 只有不可变迭代器（const_iterator），不允许修改元素的值。

map 和 set 一样也是红黑树，不同在于：map 只有键 K 的部分会参与排序，V 是个旁观者，随便修改也没关系。

所以 map 有可变迭代器，只是在其 value_type 中给 K 加上了 const 修饰：不允许修改 K，但可以修改 V。

如果你确实需要修改键值，那么请先把这个键删了，然后再以同样的 V 重新插入一遍，保证红黑树的有序。

---

map 的遍历：古代 C++98 的迭代器大法

---

map 的遍历：现代 C++17 的花哨语法糖

---

```cpp
iterator find(K const &k);
const_iterator find(K const &k) const;
```

m.find(key) 函数，根据指定的键 key 查找元素[^1]。

- 成功找到，则返回指向找到元素的迭代器
- 找不到，则返回 m.end()

第二个版本的原型作用是：如果 map 本身有 const 修饰，则返回的也是 const 迭代器。

为的是防止你在一个 const map 里 find 了以后利用迭代器变相修改 map 里的值。

[^1]: https://en.cppreference.com/w/cpp/container/map/find

---

检查过不是 m.end()，以确认成功找到后，就可以通过 * 运算符解引用获取迭代器指向的值：

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");  // 寻找 K 为 "fuck" 的元素
if (it != m.end()) {
    auto kv = *it;     // 解引用得到 K-V 对
    print(kv);         // {"fuck", 985}
    print(kv.first);   // "fuck"
    print(kv.second);  // 985
} else {
    print("找不到 fuck！");
}
```

---

检查过不是 m.end()，以确认成功找到后，就可以通过 * 运算符解引用获取迭代器指向的值：

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");  // 寻找 K 为 "fuck" 的元素
if (it != m.end()) {
    auto kv = *it;     // 解引用得到 K-V 对
    print(kv);         // {"fuck", 985}
    print(kv.first);   // "fuck"
    print(kv.second);  // 985
} else {
    print("找不到 fuck！");
}
```

---

注意 `*it` 解引用得到的是 K-V 对，需要 `(*it).second` 才能获取单独的 V。

好在 C 语言就有 `->` 运算符作为语法糖，我们可以简写成 `it->second`，与 `(*it).second` 等价。

```cpp
map<string, int> m = {
    {"fuck", 985},
};
auto it = m.find("fuck");   // 寻找 K 为 "fuck" 的元素
if (it != m.end()) {
    print(it->second);      // 迭代器有效，可以直接获得值部分 985
} else {
    print("找不到 fuck！");  // 这个分支里不得用 * 和 -> 运算符解引用 it
}
```

大多数情况下我们查询只需要获取值 V 的部分就行了，直接 `it->second` 就可以了✅

> 注意：find 找不到键时，会返回 `m.end()`，这是个无效迭代器，只作为标识符使用（类比 Python 中的 find 有时会返回 -1）。
>
> 没有确认 `it != m.end()` 前，不可以访问 `it->second`！那相当于解引用一个空指针，会造成 segfault（更专业一点说是 UB）。
>
> 记住，一定要在 `it != m.end()` 的分支里才能访问 `it->second` 哦！你得先检查过饭碗里没有老鼠💩之后，才能安心吃饭！
>
> 如果你想让老妈（标准库）自动帮你检查有没有老鼠💩，那就用会自动报错的 at（类比 Python 中的 index 找不到直接报错）。
>
> 之所以用 find，是因为有时饭碗里出老鼠💩，是计划的一部分！例如当有老鼠💩时你可以改吃别的零食。而 at 这个良心老妈呢？一发现老鼠💩就拖着你去警察局报案，零食（默认值）也不让你吃了。今日行程全部取消，维权（异常处理，找上层 try-catch 块）设为第一要务。

---

带默认值的查询

众所周知，Python 中的 dict 有一个 m.get(key, defl) 的功能，效果是当 key 不存在时，返回 defl 这个默认值代替 m[key]，而 C++ 的 map 却没有，只能用一套组合拳代替：

```cpp
m.count(key) ? m.at(key) : defl
```

但上面这样写是比较低效的，相当于查询了 map 两遍，at 里还额外做了一次多余的异常判断。

正常来说是用通用 find 去找，返回一个迭代器，然后判断是不是 end() 决定要不要采用默认值。

```cpp
auto it = m.find(key);
return it == m.end() ? it->second : defl;
```

> 饭碗里发现了老鼠💩？别急着报警，这也在我的预料之中：启用 B 计划，改吃 defl 这款美味零食即可！
>
> 如果是良心老妈 at，就直接启用 C 计划：![Plan C](images/planc.png) 抛出异常然后奔溃了，虽然这很方便我们程序员调试。

---

由于自带默认值的查询这一功能实在是太常用了，为了把这个操作浓缩到一行，我建议同学们封装成函数放到自己的项目公共头文件（一般是 utils.h 之类的名称）里方便以后使用：

```cpp
template <class M>
typename M::mapped_type map_get
( M const &m
, typename M::key_type const &key
, typename M::mapped_type const &defl
) {
  typename M::const_iterator it = m.find(key);
  if (it != m.end()) {
    return it->second;
  } else {
    return defl;
  }
}
```

```cpp
int val = map_get(config, "timeout", -1);  // 如果配置文件里不指定，则默认 timeout 为 -1
```

---

这样还不够优雅，我们还可以更优雅地运用 C++17 的函数式容器 optional：

```cpp
template <class M>
std::optional<typename M::mapped_type> map_get
( M const &m
, typename M::key_type const &key
) {
  typename M::const_iterator it = m.find(key);
  if (it != m.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}
```

当找不到时就返回 nullopt，找到就返回含有值的 optional。

> 注：本段代码已附在案例代码库的 "map_get.h" 文件中，等课后可以去 GitHub 下载。

---

调用者可以自行运用 optional 的 value_or 函数[^1]指定找不到时采用的默认值：

```cpp
int val = map_get(config, "timeout").value_or(-1);
```

如果要实现 at 同样的找不到就自动报错功能，那就改用 value 函数：

```cpp
int val = map_get(config, "timeout").value();
```

以上是典型的函数式编程范式 (FP)，C++20 还引入了更多这样的玩意[^2]，等有空会专门开节课为大家一一介绍。

```cpp
auto even = [] (int i) { return 0 == i % 2; };
auto square = [] (int i) { return i * i; };
for (int i: std::views::iota(0, 6)
          | std::views::filter(even)
          | std::views::transform(square))
    print(i);  // 0 4 16
```

[^1]: https://en.cppreference.com/w/cpp/utility/optional/value_or
[^2]: https://en.cppreference.com/w/cpp/ranges/filter_view

---

现在学习删除元素用的 erase 函数，其原型如下[^1]：

```cpp
size_t erase(K const &key);
iterator erase(iterator pos);
iterator erase(iterator beg, iterator end);
```

[^1]: https://en.cppreference.com/w/cpp/container/map/erase

---

erase 运用举例：删除一个元素

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
msg.erase("fuck");
print(msg);
```

```
{"fuck": "rust", "hello": "world"}
{"hello": "world"}
```

---

erase 的返回值和 count 一样，返回成功删除的元素个数，类型为 size_t（无符号 64 位整数）。

由于 map 中同一个键最多只可能有一个元素，取值只能为 0 或 1。

并且 size_t 可以隐式转换为 bool 类型，0 则 false，1 则 true。

因此可以直接通过 erase 的返回值是否为 0 判断是否删除成功：

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fuck", "rust"},
};
print(msg);
if (msg.erase("fuck")) {
    print("删除fuck成功");
} else {
    print("删除fuck失败");
}
if (msg.erase("suck")) {
    print("删除suck成功");
} else {
    print("删除suck失败");
}
print(msg);
```

```
{"fuck": "rust", "hello": "world"}
"删除fuck成功"
"删除suck失败"
{"hello": "world"}
```

---

一边遍历一边删除部分元素（错误示范）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
for (auto const &[k, v]: msg) {
    if (k.find("fuck") == 0) {
        msg.erase(k);  // 遍历过程中动态删除元素，会导致正在遍历中的迭代器失效，奔溃
    }
}
print(msg);
```

```
Segmentation fault (core dumped)
```

---

引出问题：迭代器失效

- 每当往 map 中插入新元素时，原先保存的迭代器不会失效。
- 删除 map 中的其他元素时，也不会失效。
- **只有当删除的刚好是迭代器指向的那个元素时，才会失效**。

```cpp
map<string, int> m = {
    {"fuck", 985},
};
m.find("fuck");
m["suck"] = 211;

```

---

一边遍历一边删除部分元素（正解[^1]）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
for (auto it = m.begin(); it != m.end(); ) {  // 没有 ++it
    auto const &[k, v] = *it;
    if (k.find("fuck") == 0) {
        it = msg.erase(it);
    } else {
        ++it;
    }
}
print(msg);
```

```
{"good": "job", "hello": "world"}
```

[^1]: https://stackoverflow.com/questions/8234779/how-to-remove-from-a-map-while-iterating-it

---

批量删除符合条件的元素（C++20[^1]）

```cpp
map<string, string> msg = {
    {"hello", "world"},
    {"fucker", "rust"},
    {"fucking", "java"},
    {"good", "job"},
};
std::erase_if(msg, [&] (auto const &kv) {
    auto &[k, v] = kv;
    return k.starts_with("fuck");
});
print(msg);
```

```
{"good": "job", "hello": "world"}
```

[^1]: https://www.apiref.com/cpp-zh/cpp/container/map/erase_if.html

---

---

接下来开始学习如何插入元素，map 的成员 insert 函数原型如下[^1]：

```cpp
pair<iterator, bool> insert(pair<const K, V> const &kv);
pair<iterator, bool> insert(pair<const K, V> &&kv);
```

他的参数类型就是刚刚介绍的 `value_type`，也就是 `pair<const K, V>`。

pair 是一个 STL 中常见的模板类型，`pair<K, V>` 有两个成员变量：

- first：V 类型，表示要插入元素的键
- second：K 类型，表示要插入元素的值

我称之为"键值对"。

[^1]: https://en.cppreference.com/w/cpp/container/map/insert

---

试着用 insert 插入键值对：

```cpp
map<string, int> m;
pair<string, int> p;
p.first = "fuck";  // 键
p.second = 985;    // 值
m.insert(p);  // pair<string, int> 可以隐式转换为 insert 参数所需的 pair<const string, int>
print(m);
```

结果：

```
{"fuck": 985}
```

---

简化 insert

<!-- <v-clicks> -->

1. 直接使用 pair 的构造函数，初始化 first 和 second

```cpp
pair<string, int> p("fuck", 985);
m.insert(p);
```

2. 不用创建一个临时变量，pair 表达式直接作为 insert 函数的参数

```cpp
m.insert(pair<string, int>("fuck", 985));
```

2. 可以用 `std::make_pair` 这个函数，自动帮你推导模板参数类型，省略 `<string, int>`

```cpp
m.insert(make_pair("fuck", 985));  // 虽然会推导为 pair<const char *, int> 但还是能隐式转换为 pair<const string, int>
```

3. 由于 insert 函数原型已知参数类型，可以直接用 C++11 的花括号初始化列表 {...}，无需指定类型

```cpp
m.insert({"fuck", 985});           // ✅
```

<!-- </v-clicks> -->

---
layout: two-cols-header
---

因此，insert 的最佳用法是：

```cpp
map<K, V> m;
m.insert({"key", "val"});
```

insert 插入和 [] 写入的异同：

- 同：当键 K 不存在时，insert 和 [] 都会创建键值对。
- 异：当键 K 已经存在时，insert 不会覆盖，默默离开；而 [] 会覆盖旧的值。

例子：

::left::

```cpp
map<string, string> m;
m.insert({"key", "old"});
m.insert({"key", "new"});  // 插入失败，默默放弃不出错
print(m);
```

```
{"key": "old"}
```

::right::

```cpp
map<string, string> m;
m["key"] = "old";
m["key"] = "new";        // 已经存在？我踏马强行覆盖！
print(m);
```

```
{"key": "new"}
```

---

insert 的返回值是 `pair<iterator, bool>` 类型，<del>STL 的尿性：在需要一次性返回两个值时喜欢用 pair</del>。

这又是一个 pair 类型，其具有两个成员：

- first：iterator 类型，是个迭代器
- second：bool 类型，表示插入成功与否，如果发生键冲突则为 false

其中 first 这个迭代器指向的是：

- 如果插入成功（second 为 true），指向刚刚成功插入的元素位置
- 如果插入失败（second 为 false），说明已经有相同的键 K 存在，发生了键冲突，指向已经存在的那个元素

---

其实 insert 返回的 first 迭代器等价于插入以后再重新用 find 找到刚刚插入的那个键，只是效率更高：

```cpp
auto it = m.insert({k, v}).first;  // 高效，只需遍历一次
```

```cpp
m.insert({k, v});     // 插入完就忘事了
auto it = m.find(k);  // 重新遍历第二次，但结果一样
```

参考 C 编程网[^1]对 insert 返回值的解释：

> 当该方法将新键值对成功添加到容器中时，返回的迭代器指向新添加的键值对；
>
> 反之，如果添加失败，该迭代器指向的是容器中和要添加键值对键相同的那个键值对。

[^1]: http://c.biancheng.net/view/7241.html

---

可以用 insert 返回的 second 判断插入多次是否成功：

```cpp
map<string, string> m;
print(m.insert({"key", "old"}).second);  // true
print(m.insert({"key", "new"}).second);  // false
m.erase("key");     // 把原来的 {"key", "old"} 删了
print(m.insert({"key", "new"}).second);  // true
```

也可以用 structual-binding 语法拆解他返回的 `pair<iterator, bool>`：

```cpp
map<string, int> counter;
auto [it, success] = counter.insert("key", 1);  // 直接用
if (!success) {  // 如果已经存在，则修改其值+1
    it->second = it->second + 1;
} else {  // 如果不存在，则打印以下信息
    print("created a new entry!");
}
```

以上这一长串代码和之前“优雅”的计数 [] 等价：

```cpp
counter["key"]++;
```

---

在 C++17 中，[] 写入有了个更高效的替代品 insert_or_assign[^1]：

```cpp
pair<iterator, bool> insert_or_assign(K const &k, V v);
pair<iterator, bool> insert_or_assign(K &&k, V v);
```

正如他名字的含义，“插入或者写入”：

- 如果 K 不存在则创建（插入）
- 如果 K 已经存在则覆盖（写入）

用法如下：

```cpp
m.insert_or_assign("key", "new");  // 与 insert 不同，他不需要 {...}，他的参数就是两个单独的 K 和 V
```

返回值依旧是 `pair<iterator, bool>`。由于这函数在键冲突时会覆盖，按理说是必定成功了，因此这个 bool 的含义从“是否插入成功”变为“是否创建了元素”，如果是创建的新元素返回true，如果覆盖了旧元素返回false。

[^1]: https://en.cppreference.com/w/cpp/container/map/insert_or_assign

---

看来 insert_or_assign 和 [] 的效果完全相同！都是在键值冲突时覆盖旧值。

既然 [] 已经可以做到同样的效果，为什么还要发明个 insert_or_assign 呢？

insert_or_assign 的优点是**不需要调用默认构造函数**，可以提升性能。

其应用场景有以下三种情况：

- ⏱ 您特别在乎性能
- ❌ 有时 V 类型没有默认构造函数，用 [] 编译器会报错
- 🥵 强迫症发作

否则用 [] 写入也是没问题的。

insert_or_assign 能取代 [] 的岗位仅限于纯写入，之前 `counter[key]++` 这种“优雅”写法依然是需要用 [] 的。

---
layout: two-cols-header
---

创建新键时，insert_or_assign 更高效。

::left::

```cpp
map<string, string> m;
m["key"] = "old";
m["key"] = "new";
print(m);
```

```
{"key": "new"}
```

覆盖旧键时，使用 [] 造成的开销：

- 调用移动赋值函数 `V &operator=(V &&)`

创建新键时，使用 [] 造成的开销：

- 调用默认构造函数 `V()`
- 调用移动赋值函数 `V &operator=(V &&)`

::right::

```cpp
map<string, string> m;
m.insert_or_assign("key", "old");
m.insert_or_assign("key", "new");
print(m);
```

```
{"key": "new"}
```

覆盖旧键时，使用 insert_or_assign 造成的开销：

- 调用移动赋值函数 `V &operator=(V &&)`

创建新键时，使用 insert_or_assign 造成的开销：

- 调用移动构造函数 `V(V &&)`

---

如果你有性能强迫症，并且是 C++17 标准：

- 写入用 insert_or_assign
- 读取用 at

如果没有性能强迫症，或者你的编译器不支持 C++17 标准：

- 写入用 []
- 读取用 at

最后，如果你是还原论者，只需要 find 和 insert 函数就是完备的了，别的函数都不用去记。所有 at、[]、insert_or_assign 之类的操作都可以通过 find 和 insert 的组合拳实现，例如刚刚我们自定义的 map_get。

---

刚刚介绍的那些 insert 一次只能插入一个元素，insert 还有一个特殊的版本，用于批量插入一系列元素。

```cpp
template <class InputIt>
void insert(InputIt beg, InputIt end);
```

参数[^1]是两个迭代器 beg 和 end，组成一个区间，之间是你要插入的数据。

该区间可以是任何其他容器的 begin() 和 end() 迭代器——那会把该容器中所有的元素都插入到本 map 中去。

例如，把 vector 中的键值对批量插入 map：

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
};
map<string, int> config;
config.insert(kvs.begin(), kvs.end());
print(config);  // {"delay": 211, "timeout": 985}
```

[^1]: 轶事：在标准库文档里批量插入版 insert 函数的 beg 和 end 这两个参数名为 first 和 last，但与 pair 的 first 并没有任何关系，只是为了防止变量名和 begin() 和 end() 成员函数发生命名冲突。为了防止同学们与 pair 的 first 混淆所以才改成 beg 和 end 做参数名。

---

注：由于 insert 不覆盖的特性，如果 vector 中有重复的键，则会以键第一次出现时的值为准，之后重复出现的键会被忽视。

```cpp
vector<pair<string, int>> kvs = {
    {"timeout", 985},
    {"delay", 211},
    {"delay", 666},
    {"delay", 233},
    {"timeout", 996},
};
map<string, int> config;
config.insert(kvs.begin(), kvs.end());
print(config);
```

```
{"delay": 211, "timeout": 985}
```

---

批量 insert 运用案例：两个 map 合并

这个批量 insert 输入的迭代器可以是任何容器，甚至可以是另一个 map 容器。

运用这一点可以实现两个 map 的并集操作。

```cpp
map<string, int> m1 = {  // 第一个 map
    {"answer", 42},
    {"timeout", 7},
};
map<string, int> m2 = {  // 第二个 map
    {"timeout", 985},
    {"delay", 211},
};
m1.insert(m2.begin(), m2.end());  // 把 m2 的内容与 m1 合并，结果写回到 m1
print(m1);
```

```
{"answer": 42, "delay": 211, "timeout": 7}
```

注：还是由于 insert 不覆盖的特性，当遇到重复的键时（例如上面的 "timeout"），会以 m1 中的值为准。

---

使用 `m1.insert(m2.begin(), m2.end())` 后，合并的结果会就地写入 m1。

如果希望合并结果放到一个新的 map 容器中而不是就地修改 m1，请自行生成一份 m1 的深拷贝：

```cpp
const map<string, int> m1 = {  // 第一个 map，修饰有 const 禁止修改
    {"answer", 42},
    {"timeout", 7},
};
const map<string, int> m2 = {  // 第二个 map，修饰有 const 禁止修改
    {"timeout", 985},
    {"delay", 211},
};
auto m12 = m1;  // 生成一份 m1 的深拷贝 m12，避免 insert 就地修改 m1
m12.insert(m2.begin(), m2.end());
print(m12);     // m1 和 m2 的合并结果
```

```
{"answer": 42, "delay": 211, "timeout": 7}
```

---

```cpp
auto m12 = m1;
m12.insert(m2.begin(), m2.end());
print(m12);     // m1 和 m2 的合并结果，键冲突时优先取 m1 的值
```

```
{"answer": 42, "delay": 211, "timeout": 7}
```

刚刚写的 m1 和 m2 合并，遇到重复时会优先采取 m1 里的值，如果希望优先采取 m2 的呢？反一反就可以了：

```cpp
auto m12 = m2;
m12.insert(m1.begin(), m1.end());
print(m12);     // m1 和 m2 的合并结果，键冲突时优先取 m2 的值
```

```
{"answer": 42, "delay": 211, "timeout": 985}
```

要是不会反，那手写一个 for 循环遍历 m2，然后 m1.insert_or_assign(k2, v2) 也是可以的，总之要懂得变通。

---

有同学就问了，这个 insert 实现了 map 的并集操作，那交集操作呢？这其实是 set 的常规操作而不是 map 的：

- set_intersection（取集合交集）
- set_union（取集合并集）
- set_difference（取集合差集）
- set_symmetric_difference（取集合对称差集）

非常抱歉在之前的 set 课中完全没有提及，因为我认为那是 `<algorithm>` 头文件里的东西。

不过别担心，之后我们会专门有一节 algorithm 课详解 STL 中这些全局函数——我称之为算法模板，因为他提供了很多常用的算法，对小彭老师这种算法弱鸡而言，实在非常好用，妈妈再也不用担心我的 ACM 奖杯。

在小彭老师制作完 algorithm 课之前，同学们可以自行参考 https://blog.csdn.net/u013095333/article/details/89322501 提前进行学习。

```cpp
std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::inserter(C, C.begin()));  // C = A U B
```

> 注：set_union 仅仅要求输入的两个区间有序，可以是 set，也可以是排序过的 vector，通过重载运算符或者指定 compare 函数，可以模拟 map 只对 key 部分排序的效果——参考 thrust::sort_by_key，但很可惜 STL 没有，需要自定义 compare 函数模拟。

---

C++11 还引入了一个以初始化列表（initializer_list）为参数的版本：

```cpp
void insert(initializer_list<pair<K, V>> ilist);
```

用法和 map 的构造函数一样，还是用花括号列表：

```cpp
map<string, int> m = {  // 初始化时就插入两个元素
    {"answer", 42},
    {"timeout", 7},
};
m.insert({              // 批量再插入两个新元素
    {"timeout", 985},   // "timeout" 发生键冲突，根据 insert 的特性，不会覆盖
    {"delay", 211},
});
```

```
{"answer": 42, "delay": 211, "timeout": 7}
```

注：这里还是和 insert({k, v}) 一样的特性，重复的键 "timeout" 没有被覆盖，依旧了保留原值。

---

小彭老师锐评批量 insert

```cpp
m.insert({
    {"timeout", 985},
    {"delay", 211},
});
```

总之这玩意和分别调用两次 insert 等价，也并不高效多少：

```cpp
m.insert({"timeout", 985});
m.insert({"delay", 211});
```

如果需要覆盖原值的批量插入，还是乖乖用 for 循环调用 [] 或 insert_or_assign 吧。

> 问：既然和批量插入没什么区别，那批量 insert 究竟还有什么存在的必要呢？
>
> map 又不是 vector 因为一个个分别插入会变成 $O(n^2)$ 复杂度，确实需要提供个批量插入的方法。
>
> 答：是为了统一，既然 vector 都有批量 insert，那 set 和 map 也得有才符合完美主义美学，而且用他来合并两个 map 也很方便。

---

类似于批量 insert，还有一个 assign 函数，作用是清空原有内容，直接设为一个全新的 map：

```cpp
map<string, int> m = {  // 初始化时就插入两个元素
    {"answer", 42},
    {"timeout", 7},
};
m.assign({              // 原有内容全部清空！重新插入两个新元素
    {"timeout", 985},
    {"delay", 211},
});
```

```
{"delay": 211, "timeout": 985}
```

注意这不是批量版 insert_or_assign，原有的 "answer" 键也被删掉了。所以搞了半天这个 assign 就是和 = 等价：
```cpp
m = {   // 等价于 assign
    {"timeout", 985},
    {"delay", 211},
};
```

---

> 那么 assign 究竟有什么存在的必要呢？是不是 $\beta$ 格稍微高一点？小彭老师也不明白，为什么标准库要发明这么多功能一样的同义词函数。

还是有必要的，虽然接受花括号列表这个版本和 = 没区别，但 assign 还有个接收两个迭代器作为输入区间的版本：

```cpp
m.assign(v.begin(), v.end());
```

等于号 = 就做不到接受两个参数，但还是可以用构造函数 + 移动赋值函数的组合拳平替，而且更易读：

```cpp
m = map<string, int>(v.begin(), v.end());
```

> 注：在 C++98 中由于没有移动语义，以上写法会调用拷贝赋值函数，造成额外一次不必要的拷贝，当时确实需要 assign 才保证高效。

---

```cpp
iterator insert(const_iterator pos, pair<K, V> const &kv);
```

这又是 insert 函数的一个重载版，增加了 pos 参数提示插入位置，我的评价是[意义不明的存在](https://www.bilibili.com/video/av9609348?p=1)，官方文档称[^1]：

> Inserts value in the position as close as possible to the position just prior to pos.
>
> 把元素（键值对）插入到位于 pos 之前，又离 pos 尽可能近的地方。

然而 map 作为红黑树应该始终保持有序，插入位置可以由 K 唯一确定，为啥还要提示？C 编程网的说法是[^2]：

> （带提示的 insert 版本）中传入的迭代器，仅是给 map 容器提供一个建议，并不一定会被容器采纳。该迭代器表明将新键值对添加到容器中的位置。需要注意的是，新键值对添加到容器中的位置，并不是此迭代器说了算，最终仍取决于该键值对的键的值。

也就是说这玩意还不一定管用，只是提示性质的（和 mmap 函数的 start 参数很像，你可以指定，但只是个提示，指定了不一定有什么软用，具体什么地址还是操作系统说了算，他从返回值里给你的地址才是正确答案）。

估计又是性能强迫症发作了吧，例如已知指向 "key" 的迭代器，想要插入 "kea"，那么指定指向 "key" 的迭代器就会让 find 能更容易定位到 "kea" 要插入的位置？如果你知道准确原因和运用案例，欢迎在评论区指出。

[^1]: https://en.cppreference.com/w/cpp/container/map/insert
[^2]: http://c.biancheng.net/view/7241.html

---

小彭老师 锐评 分奴

玩游戏的人当中，分奴指的是那些打分看的很重的人，为了上分、胜利，可以不择手段。

而玩编程的人当中，有一种性能强迫症的现象，特点有：

1. 他们把一点鸡毛蒜皮的性能看得很重，为了 1% 的提升他们可以放弃可维护性，可移植性，可读性
2. 对着不是瓶颈的冷代码一通操作猛如虎，结果性能瓶颈根本不是这儿，反而瓶颈部分的热代码他们看不到
3. 根本没有进行性能测试（profling）就在那焦虑不存在的性能瓶颈，杞人忧天，妄下结论
4. 根本不具备优化的经验，对计算机组成原理理解粗浅，缺乏常识，认为“执行的指令数量多少”就决定了性能
5. 不以性能测试结果为导向，自以为是地心理作用优化，结果性能没有提升，反而优化出一堆 bug
6. 对于并行的期望过高，以为并行是免费的性能提升，根本不明白并行需要对程序算法进行多少破坏性的改动
7. 知小礼而无大义，一边执着地问我“如何用OpenMP并行”，一边还在“并行地做strcmp”，相当于金箔擦屁股。
8. 只看到常数优化的作用，“把 Python 换成 C++ 会不会好一点啊”，然而他们给我一看代码，用了 list.index，复杂度都是 $O(N^2)$ 的，即使换成 C++ 用天河二号跑也无非是从小垃圾变成大垃圾（真实案例）。

我称之为编程界的分奴。

---

insert 的究极分奴版：emplace

```cpp
template <class Args>
pair<iterator, bool> emplace(Args &&...args);
```

虽然变长参数列表 `Args &&...args` 看起来很酷，然而由于 map 的特殊性，其元素类型是 `pair<const K, V>`，而 pair 的构造函数只有两个参数，导致实际上这个看似炫酷的变长参数列表往往只能接受两个参数，因此这个函数的调用方法实际上只能是：

```cpp
pair<iterator, bool> emplace(K k, V v);
```

写法：

```cpp
m.emplace(key, val);
```

等价于：

```cpp
m.insert({key, val});
```

我的评价是：emplace 对于 set，如果元素类型是 `array<int, 100>` 这样比较大的类型，确实能起到减少移动构造函数的作用，但是这个 map 他的元素类型不是直接的 V 而是一个 pair，他分的是 pair 的构造函数，没有用，V 部分还是会造成一次额外的移动开销，所以这玩意除了妨碍安全性，可读性以外，没有任何收益的，不建议 map 使用 emplace（set 和 vector 可以用 emplace）。

返回值还是 `pair<iterator, bool>`，其意义和 insert 一样，不再赘述。

---

insert 的宇宙无敌分奴版：emplace_hint

```cpp
template <class Args>
pair<iterator, bool> emplace_hint(const_iterator pos, Args &&...args);
```

写法：

```cpp
m.emplace_hint(pos, key, val);
```

等价于：

```cpp
m.insert(pos, {key, val});
```

之所以要分两个函数名 emplace 和 emplace_hint 而不是利用重载区分，是因为直接传入 pos 会被 emplace 当做 pair 的构造参数，而不是插入位置提示。

- emplace 对应于普通的 `insert(pair<const K, V>)` 这一重载。
- emplace_hint 对应于带插入位置提示的 `insert(const_iterator, pair<const K, V>)` 这一重载。

由于带提示的 insert 本来就意义不大，加上 emplace 追求性能意义不大，所以这个函数是真的没必要用。

---

insert 的托马斯黄金大回旋分奴版：try_emplace

```cpp
template <class Args>
pair<iterator, bool> try_emplace(K const &k, Args &&...args);
```

写法：

```cpp
m.try_emplace(key, arg1, arg2, ...);
```

等价于：

```cpp
m.insert({key, V(arg1, arg2, ...)});
```

由于 emplace 实在是憨憨，他变长参数列表就地构造的是 pair，然而 pair 的构造函数正常不就是只有两个参数吗，变长没有用。实际有用的往往是我们希望用变长参数列表就地构造值类型 V，对 K 部分并不关系。因此 C++17 引入了 try_emplace，其键部分保持 `K const &`，值部分采用变长参数列表。

我的评价是：这个比 emplace 实用多了，如果要与 vector 的 emplace_back 对标，那么 map 与之对应的一定是 try_emplace。同学们如果要分奴的话还是建议用 try_emplace。

---

insert 的炫彩中二摇摆混沌大魔王分奴版：带插入位置提示的 try_emplace

写法：

```cpp
m.try_emplace(pos, key, arg1, arg2, ...);
```

等价于：

```cpp
m.insert(pos, {key, V(arg1, arg2, ...)});
```

> 之所以不需要再分一个 try_emplace_hint，是因为 try_emplace 的第一个参数是 K 类型，不可能和 const_iterator 类型混淆，因此 C++ 委员会最终决定直接用同一个名字，让编译器自动重载了。

还是一如既往的意义不大。

---

# 增删

| 写法 | 效果 | 版本 | 推荐 |
|------|------|------|------|
| `m.insert(make_pair(key, val))` | 插入但不覆盖 | C++98 | 💩 |
| `m.insert({key, val})` | 插入但不覆盖 | C++11 | ❤ |
| `m.emplace(key, val)` | 插入但不覆盖 | C++11 | 💩 |
| `m.try_emplace(key, valcons...)` | 插入但不覆盖 | C++17 | 💣 |
| `m.insert_or_assign(key, val)` | 插入或覆盖 | C++17 | ❤ |
| `m[key] = val` | 插入或覆盖 | C++98 | 💣 |
| `m.erase(key)` | 删除指定元素 | C++98 | ❤ |

---

# 改查

| 写法 | 效果 | 版本 | 推荐 |
|------|------|------|------|
| `m.at(key)` | 找不到则出错，找到则返回引用 | C++98 | ❤ |
| `m[key]` | 找不到则自动创建`0`值，返回引用 | C++98 | 💣 |
| `myutils::map_get(m, key, defl)` | 找不到则返回默认值 | C++98 | ❤ |
| `m.find(key) == m.end()` | 检查键 `key` 是否存在 | C++98 | 💣 |
| `m.count(key)` | 检查键 `key` 是否存在 | C++98 | ❤ |
| `m.contains(key)` | 检查键 `key` 是否存在 | C++20 | 💩 |

---

# 初始化

| 写法 | 效果 | 版本 | 推荐 |
|------|------|------|------|
| `map<K, V> m = {{k1, v1}, {k2, v2}}` | 初始化为一系列键值对 | C++11 | ❤ |
| `auto m = map<K, V>{{k1, v1}, {k2, v2}}` | 初始化为一系列键值对 | C++11 | 💩 |
| `func({{k1, v1}, {k2, v2}})` | 给函数参数传入一个 map | C++11 | ❤ |
| `m = {{k1, v1}, {k2, v2}}` | 重置为一系列键值对 | C++11 | ❤ |
| `m.clear()` | 清空所有表项 | C++98 | ❤ |
| `m = {}` | 清空所有表项 | C++11 | 💣 |

---

swap 与 move

---

extract 和 merge 函数

---

时间复杂度问题

---

允许重复键值的 multimap

---

| 函数或写法 | 解释说明 | 时间复杂度 |
|-|-|-|
| m1 = move(m2) | 移动 | $O(1)$ |
| m1 = m2 | 拷贝 | $O(N)$ |
| swap(m1, m2) | 交换 | $O(1)$ |
| m.clear() | 清空 | $O(N)$ |

---

| 函数或写法 | 解释说明 | 时间复杂度 |
|-|-|-|
| m.insert({key, val}) | 插入键值对 | $O(\log N)$ |
| m.insert(pos, {key, val}) | 带提示的插入，如果位置提示准确 | $O(1)$+ |
| m.insert(pos, {key, val}) | 带提示的插入，如果位置提示不准确 | $O(\log N)$ |
| m[key] = val | 插入或覆盖 | $O(\log N)$ |
| m.insert_or_assign(key, val) | 插入或覆盖 | $O(\log N)$ |
| m.insert({vals...}) | 设 M 为待插入元素（vals）的数量 | $O(M \log N)$ |
| map m = {vals...} | 如果 vals 无序 | $O(N \log N)$ |
| map m = {vals...} | 如果 vals 已事先从小到大排列 | $O(N)$ |

---

| 函数或写法 | 解释说明 | 时间复杂度 |
|-|-|-|
| m.at(key) | 根据指定的键，查找元素，返回值的引用 | $O(\log N)$ |
| m.find(key) | 根据指定的键，查找元素，返回迭代器 | $O(\log N)$ |
| m.count(key) | 判断是否存在指定键元素，返回相同键的元素数量（只能为 0 或 1） | $O(\log N)$ |
| m.equal_range(key) | 根据指定的键，确定上下界，返回区间 | $O(\log N)$ |
| m.size() | map 中所有元素的数量 | $O(1)$ |
| m.erase(key) | 根据指定的键，删除元素 | $O(\log N)$ |
| m.erase(it) | 根据找到的迭代器，删除元素 | $O(1)+$ |
| m.erase(beg, end) | 批量删除区间内的元素，设该区间（beg 和 end 之间）有 M 个元素 | $O(M + \log N)$ |
| erase_if(m, cond) | 批量删除所有符合条件的元素 | $O(N)$ |

---

| 函数或写法 | 解释说明 | 时间复杂度 |
|-|-|-|
| m.insert(node) | | $O(\log N)$ |
| node = m.extract(it) | | $O(1)+$ |
| node = m.extract(key) | | $O(\log N)$ |
| m1.merge(m2) | 合并两个 map，清空 m2，结果写入 m1 | $O(\log N)$ |
| m1.insert(m2.begin(), m2.end()) | 合并两个 map，m2 保持不变，结果写入 m1 | $O(N \log N)$ |

---

基于哈希散列的映射表 unordered_map

用法上，unordered_map 基本与 map 相同，这里着重介绍他们的不同点。

---

区别 1：有序性

- map 基于红黑树，元素从小到大顺序排列，遍历时也是从小到大的，键类型需要支持比大小（std::less）。
- unordered_map 基于哈希散列表，里面元素顺序随机，键类型需要支持哈希值计算（std::hash）。

map 中的元素始终保持有序，unordered_map 里面的元素是随机的。

这也意味着 std::set_union 这类要求输入区间有序的 algorithm 函数无法适用于 unordered_map/set。

---

区别 2：时间复杂度

- map 的查询和插入操作是 $O(\log N)$ 复杂度的。
- unordered_map 的查询和插入操作是 $O(1)$ 复杂度的。

处理很高的数据量时，unordered_map 更高效。

但 unordered_map 需要频繁地进行 rehash 操作保持高效，否则不如 map。

---

区别 3：迭代器失效条件

- map 和 unordered_map 都是只有当删除的刚好是迭代器指向的那个元素时才会失效，这点相同。
- 但 unordered_map 的 rehash 操作（需要经常 rehash 以提升性能）会造成所有迭代器失效。

unordered_map 不会自动 rehash，rehash 需要手动调用，因此通常来说不必担心 unordered_map 的迭代器失效。

---

```cpp
```

区别于基于红黑树的映射表 map

---

自定义比较器 / 哈希函数

---

map 中的 RAII

---

map 和 unique_ptr 结合使用

---

map 和 function 结合使用

---

案例：全局句柄表实现仿 C 语言 API

---

案例：全局注册表实现动态反射
