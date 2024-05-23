# 回调函数·破局·设计模式

设计模式大都是在**面向对象**时代提出的，诚然，有一部分模式，天生适合面向对象的写法。

但是其中有一些写起来非常复杂而难懂，实际上是因为面向对象范式的固有缺陷。

- 策略模式
- 观察者模式
- 访问者模式
- 装饰器模式
- 命令模式
- 工厂模式
- 过滤器模式
- 构建者模式
- 代理模式

后来出现的**函数式编程范式**大大简化了这些。他创新性地引入了大名鼎鼎的**回调函数** (Delegate) 与**闭包** (Closure)。

古代 Java 是因为不支持函数式编程范式，只支持面向对象范式，才不得不用原始的繁琐的面向对象写法。

可恶的是，很多古板的所谓“设计模式”教程，有的甚至打着“现代 C++ 设计模式”的名号，还依然在以落后的**面向对象写法**来讲授这些明明更适合**函数式写法**的设计模式！

为此，小彭老师专门为你制作本期课程，以简化这些本就应该简单的设计模式。

我们会把函数式的写法和面向对象的写法都讲一遍，保证让你感受到函数式编程的敏捷魅力。

# 策略模式？

```cpp
void feedDog() {
    puts("喂食");      // 重复的代码
    puts("汪");
    puts("喂食完毕");  // 重复的代码
}

void feedCat() {
    puts("喂食");      // 重复的代码
    puts("喵");
    puts("喂食完毕");  // 重复的代码
}

int main() {
    feedDog();
    feedCat();
}
```

出现了重复，如果未来我们添加 feedPig，还需要重写一遍那两行完全一样的代码。

如何复用代码？在昨天的面向对象课程中，我们选择了带虚函数的接口类：

```cpp
struct Pet {
    virtual void speak() = 0;
};

struct CatPet ：Pet {
    void speak() override {
        puts("喵");
    }
};

struct DogPet ：Pet {
    void speak() override {
        puts("汪");
    }
};

void feed(Pet *pet) {
    puts("喂食");
    pet->speak();
    puts("喂食完毕");
}

int main() {
    feed(new CatPet());
    feed(new DogPet());
}
```

这样，我们就实现了代码的复用。但是，当我们需要添加喂食宠物猪的功能时，我们无需修改 feed 函数，只需添加一个 PigPet 类，其重载了 speak 为猪的叫声即可，这符合**开闭原则**。

痛点：如果你经常使用面向对象搞设计模式，会发现经常出现这种**只需要一个成员函数**的接口。以昨天课程中出现的为例：

```cpp
struct Pet {
    virtual void speak() = 0;
};

struct Inputer {
    virtual optional<int> fetch() = 0;
};

struct Reducer {
    virtual unique_ptr<ReducerState> init() = 0;
};

struct Gun {
    virtual Bullet *shoot() = 0;
};
```

每个接口都各立门户，成员函数名字各不一样，互不兼容。

写的时候还必须想两个名字，一个类名，一个函数名，非常难以复用！

## 函数指针

有没有一种可能，我们其实只需要定义一个函数，根本不用定义类？

C 语言有一种叫做函数指针的东西，可以指向函数，然后可以把这个函数指针作为参数传递给别的函数。

使用函数指针时，通常会先用 typedef 把函数指针类型起一个别名。如：

```cpp
typedef void func_t(int);
```

表示接受一个 int 做参数，返回类型是 void 的函数。给他取的别名叫 func_t。

由于 typedef 需要把类型别名的名字写在中间，有时会看不清楚。

现代 C++ 中提出了更直观的 `using =` 语法取代了落后的 typedef。

```cpp
using func_t = void (int);
```

使用时，只需要写 `func_t *` 即可：

```cpp
void hello(func_t * func);
```

这样 hello 函数就成了一个接受函数做参数的函数！

这里的 `func_t` 是函数类型，`func_t *` 就是所谓的函数指针。

**函数类型本身不能直接作为参数传递**，因此我们只能通过**函数的指针**来传递和使用函数。

> 注意，也有些教材里会直接给 func_t 的定义就加上 `*` 使其一开始就是个函数指针，这样在 hello 那里就不用写 `*` 了。

```cpp
typedef void (*funcptr_t)(int);
using funcptr_t = void (*)(int);

void hello(funcptr_t func);
```

## 开始改造

```cpp
using speaker_t = void ();

void cat_speak() {
    puts("喵");
}

void dog_speak() {
    puts("旺");
}

void feed(speaker_t * speak) {
    puts("喂食");
    speak();
    puts("喂食完毕");
}

int main() {
    feed(cat_speak);
    feed(dog_speak);
}
```

这样就实现了代码的复用！当我们需要添加喂食宠物猪的功能时，我们无需修改 feed 函数，只需添加一个 pig_speak 函数即可，同样符合**开闭原则**。

不用大费周章定义类和虚函数接口，不用 new 创建对象，更加轻量级！

## 兰姆达表达式

不仅如此，你甚至都不必大费周章跑到类外面创建 cat_speak 和 dog_speak 函数！

C++11 还引入了一款昵称为 “Lambda” 的语法糖，允许你在 main 函数内就直接创建出这两个函数。

语法就是 `[] (参数列表) -> 返回类型 { 函数体 }`。

```cpp
int main() {
    speaker_t * cat_speak = [] () -> void {
        puts("喵");
    };
    speaker_t * dog_speak = [] () -> void {
        puts("旺");
    };
    feed(cat_speak);
    feed(dog_speak);
}
```

其中返回类型的部分 `-> void` 可以省略不写。没有参数时，参数列表的括号 `()` 也可以省略不写！

而且 Lambda 可以直接写在 feed 函数的参数上，不必再大费周章给每个函数定义个名字！

```cpp
int main() {
    feed([] {
        puts("喵");
    });
    feed([] {
        puts("旺");
    });
}
```

## 无法存储状态？

```cpp
struct CatPet ：Pet {
    int age;

    CatPet(int age) : age(age) {}

    void speak() override {
        printf("喵~ 我 %d 岁了\n", age);
    }
};

int main() {
    feed(new CatPet(42));
}
```

面向对象，因为他基于类，可以很轻松地添加新变量作为对象的**状态**，例如这里添加了猫的年龄。

而函数指针，似乎无法存储状态？

C 语言中常见的做法是，额外传递一个 `void *` 指针，用于存储回调函数的状态。

```cpp
using speaker_t = void ();

void feed(speaker_t *speak, void *speak_data) {
    puts("喂食");
    speak(speak_data);
    puts("喂食完毕");
}

struct CatData {
    int age;
};

void cat_speak(void *data) {
    CatData *cat = (CatData *)data;
    printf("喵~ 我 %d 岁了\n", cat->age);
}

int main() {
    feed(cat_speak, cat_data);
}
```

而现代 C++ 思想家们提出了**仿函数对象**的概念，函数不一定就一定是函数，也可以是任意一个类！只要这个类具有 `.run()` 成员函数就可以称之为“仿函数 (functor)”。

这样，就可以把 age 这种状态变量存在“仿函数”对象体内了。

```cpp
struct CatFunctor {
    void run();
};
```
