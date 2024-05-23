# 应知应会 C++ 小技巧

## 交换两个变量

```cpp
int a = 42;
int b = 58;
```

现在你想交换这两个变量。

```cpp
int tmp = a;
a = b;
b = tmp;
```

但是标准库提供了更好的方法：

```cpp
swap(a, b);
```

这个方法可以交换任意两个同类型的值，包括结构体、数组、容器等。

## 别再写构造函数啦！

```cpp
// C++98
struct Student {
    string name;
    int age;
    int id;

    Student(string name_, int age_, int id_) : name(name_), age(age_), id(id_) {}
};

Student stu("侯捷老师", 42, 123);
```

C++98 需要手动书写构造函数，非常麻烦！而且几乎都是重复的。

C++11 中，平凡的结构体类型不需要再写构造函数了，只需用 `{}` 就能对成员依次初始化：

```cpp
// C++11
struct Student {
    string name;
    int age;
    int id;
};

Student stu{"小彭老师", 24, 123};
```

这被称为**聚合初始化** (aggergate initialize)。只要你的类没有自定义构造函数，没有 private 成员，都可以用 `{}` 聚合初始化。

C++20 中，用 `()` 也可以聚合初始化了，用起来就和传统的 C++98 构造函数一样！

```cpp
// C++20
Student stu("小彭老师", 24, 123);
```

聚合初始化还可以指定默认值：

```cpp
// C++11
struct Student {
    string name;
    int age;
    int id = 9999;
};

Student stu{"小彭老师", 24};
// 等价于：
Student stu{"小彭老师", 24, 9999};
```

C++20 开始，`{}` 聚合初始化可以根据每个成员的名字来指定值：

```cpp
Student stu{.name = "小彭老师", .age = 24, .id = 9999};
// 等价于：
Student stu{"小彭老师", 24, 9999};
```

不慎写错顺序也不用担心。

```cpp
Student stu{.name = "小彭老师", .age = 24, .id = 9999};
Student stu{.name = "小彭老师", .id = 9999, .age = 24};
```

## 别再 `[]` 啦！

你知道吗？在 map 中使用 `[]` 查找元素，如果不存在，会自动创建一个默认值。这个特性有时很方便，但如果你不小心写错了，就会在 map 中创建一个多余的默认元素。

```cpp
map<string, int> table;
table["小彭老师"] = 24;

cout << table["侯捷老师"];
```

table 中明明没有 "侯捷老师" 这个元素，但由于 `[]` 的特性，他会默认返回一个 0，不会爆任何错误！

改用更安全的 `at()` 函数，当查询的元素不存在时，会抛出异常，方便你调试：

```cpp
map<string, int> table;
table.at("小彭老师") = 24;

cout << table.at("侯捷老师");  // 抛出异常
```

`[]` 的用途是“写入新元素”时，如果元素不存在，他可以自动帮你创建一个默认值，供你以引用的方式赋值进去。

## 别再 make_pair 啦！

```cpp
map<string, int> table;
table.insert(pair<string, int>("侯捷老师", 42));
```

为避免写出类型名的麻烦，很多老师都会让你写 make_pair：

```cpp
map<string, int> table;
table.insert(make_pair("侯捷老师", 42));
```

然而 C++11 提供了更好的写法，那就是通过 `{}` 隐式构造，不用写出类型名或 make_pair：

```cpp
map<string, int> table;
table.insert({"侯捷老师", 42});
```

即使需要写出类型名的情况，也可以用 C++17 的 CTAD 语法，免去模板参数，make_xxx 系列函数就此完全被 C++17 平替：

```cpp
map<string, int> table;
table.insert(pair("侯捷老师", 42));
```

## insert 不会替换现有值

```cpp
map<string, int> table;
table.insert({"小彭老师", 24});
table.insert({"小彭老师", 42});
```

这时，`table["小彭老师"]` 仍然会是 24，而不是 42。因为 insert 不会替换 map 里已经存在的值。

如果希望如果已经存在时，替换现有元素，可以使用 `[]` 运算符：

```cpp
map<string, int> table;
table["小彭老师"] = 24;
table["小彭老师"] = 42;
```

C++17 提供了比 `[]` 运算符更适合覆盖性插入的 insert_or_assign 函数：

```cpp
map<string, int> table;
table.insert_or_assign("小彭老师", 24);
table.insert_or_assign("小彭老师", 42);
```


## cout 不需要 endl

```cpp
int a = 42;
printf("%d\n", a);
```

万一你写错了 `%` 后面的类型，编译器不会有任何报错，留下隐患。

```cpp
int a = 42;
printf("%s\n", a);  // 编译器不报错，但是运行时会崩溃！
```

C++ 中有更安全的输出方式 `cout`，通过 C++ 的重载机制，无需手动指定 `%`，自动就能推导类型。

```cpp
int a = 42;
cout << a << endl;
double d = 3.14;
cout << d << endl;
```

```cpp
cout << "Hello, World!" << endl;
```

endl 是一个特殊的流操作符，作用等价于先输出一个 `'\n'` 然后 `flush`。

```cpp
cout << "Hello, World!" << '\n';
cout.flush();
```

但实际上，输出流 cout 默认的设置就是“行刷新缓存”，也就是说，检测到 `'\n'` 时，就会自动刷新一次，根本不需要我们手动刷新！

如果还用 endl 的话，就相当于刷新了两次，浪费性能。

所以，我们只需要输出 `'\n'` 就可以了，每次换行时 cout 都会自动刷新，endl 是一个典型的以讹传讹错误写法。

```cpp
cout << "Hello, World!" << '\n';
```

## 多线程中 cout 出现乱序？

## 一边遍历 map，一边删除？

```cpp
map<string, int> table;
for (auto it = table.begin(); it != table.end(); ++it) {
    if (it->second < 0) {
        table.erase(it);
    }
}
```

会发生崩溃！看来 map 似乎不允许在遍历的过程中删除？不，只是你的写法有错误：

```cpp
map<string, int> table;
for (auto it = table.begin(); it != table.end(); ) {
    if (it->second < 0) {
        it = table.erase(it);
    } else {
        ++it;
    }
}
```

C++20 引入了更好的 erase_if 全局函数，不用手写上面这么麻烦的代码：

```cpp
map<string, int> table;
erase_if(table, [](pair<string, int> it) {
    return it.second < 0;
});
```

## 高效删除 vector 元素

```cpp
vector<int> v = {48, 23, 76, 11, 88, 63, 45, 28, 59};
```

众所周知，在 vector 中删除元素，会导致后面的所有元素向前移动，十分低效。复杂度：$O(n)$

```cpp
// 直接删除 v[3]
v.erase(v.begin() + 3);
```

如果不在乎元素的顺序，可以把要删除的元素和最后一个元素 swap，然后 pop_back。复杂度：$O(1)$

```cpp
// 把 v[3] 和 v[v.size() - 1] 位置对调
swap(v[3], v[v.size() - 1]);
// 然后删除 v[v.size() - 1]
v.pop_back();
```

这样就不用移动一大堆元素了。这被称为 back-swap-erase。

## 批量删除 vector 元素

vector 中只删除一个元素需要 $O(n)$。如果一边遍历，一边删除多个符合条件的元素，就需要复杂度 $O(n^2)$ 了。

标准库提供了 `remove` 和 `remove_if` 函数，其内部采用类似 back-swap-erase 的方法，先把要删除的元素移动到末尾。然后一次性 `erase` 掉末尾同样数量的元素。

且他们都能保持顺序不变。

删除所有值为 42 的元素：

```cpp
vector<int> v;
v.erase(remove(v.begin(), v.end(), 42), v.end());
```

删除所有值大于 0 的元素：

```cpp
vector<int> v;
v.erase(remove_if(v.begin(), v.end(), [](int x) {
    return x > 0;
}), v.end());
```

现在 C++20 也引入了全局函数 erase 和 erase_if，使用起来更加直观：

```cpp
vector<int> v;
erase(v, 42);       // 删除所有值为 42 的元素
erase_if(v, [](int x) {
    return x > 0;   // 删除所有值大于 0 的元素
});
```

## 有序的 vector

如果你想要维护一个有序的数组，用 `lower_bound` 或 `upper_bound` 来插入元素，保证插入后仍保持有序：

```cpp
vector<int> s;
s.push_back(1);
s.push_back(2);
s.push_back(4);
s.push_back(6);
// s = { 1, 2, 4, 6 }
s.insert(lower_bound(s.begin(), s.end(), 3), 3);
// s = { 1, 2, 3, 4, 6 }
s.insert(lower_bound(s.begin(), s.end(), 5), 5);
// s = { 1, 2, 3, 4, 5, 6 }
```

有序数组中，可以利用 `lower_bound` 或 `upper_bound` 快速二分查找到想要的值：

```cpp
vector<int> s;
s.push_back(1);
s.push_back(2);
s.push_back(4);
s.push_back(6);
// s = { 1, 2, 4, 6 }
lower_bound(s.begin(), s.end(), 3); // s.begin() + 2;
lower_bound(s.begin(), s.end(), 5); // s.begin() + 3;
```

利用 CDF 积分 + 二分法可以实现生成任意指定分布的随机数。

例如抽卡概率要求：

- 2% 出金卡
- 10% 出蓝卡
- 80% 出白卡
- 8% 出答辩

```cpp
vector<double> probs = {0.02, 0.1, 0.8, 0.08};
vector<double> cdf = {0.02, 0.12, 0.92, 1.00};
vector<string> result = {"金卡", "蓝卡", "白卡", "答辩"};
// 生成 100 个随机数：
for (int i = 0; i < 100; ++i) {
    double r = rand() / (RAND_MAX + 1.);
    int index = lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
    cout << "你抽到了" << result[index] << endl;
}
```

## 提前返回

```cpp
void babysitter(Baby *baby) {
    if (baby->is_alive()) {
        puts("宝宝已经去世了");
    } else {
        puts("正在检查宝宝喂食情况...");
        if (baby->is_feeded()) {
            puts("宝宝已经喂食过了");
        } else {
            puts("正在喂食宝宝...");
            puts("正在调教宝宝...");
            puts("正在安抚宝宝...");
        }
    }
}
```

这个函数有很多层嵌套，很不美观。用**提前返回**的写法来优化：

```cpp
void babysitter(Baby *baby) {
    if (baby->is_alive()) {
        puts("宝宝已经去世了");
        return;
    }
    puts("正在检查宝宝喂食情况...");
    if (baby->is_feeded()) {
        puts("宝宝已经喂食过了");
        return;
    }
    puts("正在喂食宝宝...");
    puts("正在调教宝宝...");
    puts("正在安抚宝宝...");
}
```

## 立即调用的 Lambda

有时，需要在一个列表里循环查找某样东西，也可以用提前返回的写法优化：

```cpp
bool find(const vector<int> &v, int target) {
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] == target) {
            return true;
        }
    }
    return false;
}
```

可以包裹一个立即调用的 Lambda 块 `[&] { ... } ()`，限制提前返回的范围：

```cpp
void find(const vector<int> &v, int target) {
    bool found = [&] {
        for (int i = 0; i < v.size(); ++i) {
            if (v[i] == target) {
                return true;
            }
        }
        return false;
    } ();
    if (found) {
        ...
    }
}
```

## Lambda 复用代码

```cpp
vector<string> spilt(string str) {
    vector<string> list;
    string last;
    for (char c: str) {
        if (c == ' ') {
            list.push_back(last);
            last.clear();
        } else {
            last.push_back(c);
        }
    }
    list.push_back(last);
    return list;
}
```

上面的代码可以用 Lambda 复用：

```cpp
vector<string> spilt(string str) {
    vector<string> list;
    string last;
    auto push = [&] {
        list.push_back(last);
        last.clear();
    };
    for (char c: str) {
        if (c == ' ') {
            push();
        } else {
            last.push_back(c);
        }
    }
    push();
    return list;
}
```

## 类内静态成员

在头文件中定义结构体的 static 成员时：

```cpp
struct Class {
    static Class instance;
};
```

会报错 `undefined reference to 'Class::instance'`。这是说的你需要找个 .cpp 文件，写出 `Class Class::instance` 才能消除该错误。

C++17 中，只需加个 `inline` 就能解决。

```cpp
struct Class {
    inline static Class instance;
};
```

## const 居然应该后置...

```cpp
const int *p;
int *const p;
```

你能看出来上面这个 const 分别修饰的是谁吗？

1. 指针本身 `int *`
2. 指针指向的 `int`

```cpp
const int *p;  // 2
int *const p;  // 1
```

为了看起来更加明确，我通常都会后置所有的 const 修饰。

```cpp
int const *p;
int *const p;
```

这样就一目了然，const 总是在修饰他前面的东西，而不是后面。

为什么 `int *const` 修饰的是 `int *` 也就很容易理解了。

```cpp
int const i;
int const *p;
int const &r;
```
