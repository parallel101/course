# 让虚函数再次伟大！

许多设计模式都与虚函数息息相关，今天我们来学习一些常用的。

- 策略模式
- 迭代器模式
- 适配器模式
- 工厂模式
- 超级工厂模式
- 享元模式
- 代理模式

很多教材中都会举出这种看起来好像很有说服力的例子：

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

int main() {
    Pet *cat = new CatPet();
    Pet *dog = new DogPet();
    cat->speak();
    dog->speak();
}
```

然而，在这个案例中，虚函数可有可无，并没有发挥任何价值，因为普通成员函数也可以实现同样效果。

虚函数真正的价值在于，作为一个参数传入其他函数时！可以复用那个函数里的代码。

```cpp
void feed(Pet *pet) {
    puts("喂食");
    pet->speak();
    puts("喂食完毕");
}

int main() {
    Pet *cat = new CatPet();
    Pet *dog = new DogPet();
    feed(cat);
    feed(dog);
}
```

优点在于，feed 函数只用实现一遍了。如果没有虚函数：

```cpp
void feed(DogPet *pet) {
    puts("喂食");      // 重复的代码
    puts("汪");
    puts("喂食完毕");  // 重复的代码
}

void feed(CatPet *pet) {
    puts("喂食");      // 重复的代码
    puts("喵");
    puts("喂食完毕");  // 重复的代码
}
```

`喂食` 和 `喂食完毕` 重复两遍！如果我们又要引入一种新动物 `PigPet` 呢？你又要手忙脚乱复制粘贴一份新的 feed 函数！

```cpp
void feed(PigPet *pet) {
    puts("喂食");      // 重复的代码
    puts("拱");
    puts("喂食完毕");  // 重复的代码
}
```

现在，老板突然改了需求，他说动物现在要叫两次。
采用了虚函数的你，只需要在 feed 函数内增加一次 speak 即可，轻松！

```cpp
void feed(Pet *pet) {
    puts("喂食");
    pet->speak();
    pet->speak();  // 加这里
    puts("喂食完毕");
}
```

而如果一开始没用虚函数，就得连改 3 个地方！

```cpp
void feed(DogPet *pet) {
    puts("喂食");
    puts("汪");  // 改这里
    puts("汪");  // 改这里
    puts("喂食完毕");
}

void feed(CatPet *pet) {
    puts("喂食");
    puts("喵");  // 改这里
    puts("喵");  // 改这里
    puts("喂食完毕");
}

void feed(PigPet *pet) {
    puts("喂食");
    puts("拱");  // 改这里
    puts("拱");  // 改这里
    puts("喂食完毕");
}
```

而且万一复制粘贴的时候有个地方写错了，非常隐蔽，很容易发现不了：

```cpp
void feed(PigPet *pet) {
    puts("喂食");
    puts("拱");
    puts("喵");  // 把猫的代码复制过来的时候漏改了 🤯
    puts("喂食完毕");
}
```

## 虚函数实战案例

小彭老师，你说的这些我都会，这有什么稀奇的。那我们来举个实际开发中会遇到的例子。

这里有一个求和函数，可以计算一个数组中所有数字的和。
还有一个求积函数，可以计算一个数组中所有数字的积。

```cpp
int sum(vector<int> v) {
    int res = 0;
    for (int i = 0; i < v.size(); i++) {
        res = res + v[i];
    }
    return res;
}

int product(vector<int> v) {
    int res = 1;
    for (int i = 0; i < v.size(); i++) {
        res = res * v[i];
    }
    return res;
}
```

注意到这里面有很多代码重复！

我们观察一下 sum 和 product 之间有哪些相似的部分，把两者产生不同的部分用 ??? 代替。

```cpp
int reduce(vector<int> v) {
    int res = ???;            // sum 时这里是 0，product 时这里是 1
    for (int i = 0; i < v.size(); i++) {
        res = res ??? v[i];   // sum 时这里是 +，product 时这里是 *
    }
    return res;
}
```

把 ??? 部分用一个虚函数顶替：

```cpp
struct Reducer {
    virtual int init() = 0;
    virtual int add(int a, int b) = 0;
};

int reduce(vector<int> v, Reducer *reducer) {
    int res = reducer.init();
    for (int i = 0; i < v.size(); i++) {
        res = reducer.add(res, v[i]);
    }
    return res;
}
```

这样不论我们想要求和，还是求积，只需要实现其中不同的部分就可以了，公共部分已经在 reduce 里实现好，就实现了代码复用。

```cpp
struct SumReducer : Reducer {
    int init() override {
        return 0;
    }

    int add(int a, int b) override {
        return a + b;
    }
};

struct ProductReducer : Reducer {
    int init() override {
        return 1;
    }

    int add(int a, int b) override {
        return a * b;
    }
};
```

```cpp
reduce(v, new SumReducer());     // 等价于之前的 sum(v)
reduce(v, new ProductReducer()); // 等价于之前的 product(v)
```

这就是所谓的**策略模式**。

很容易添加新的策略进来：

```cpp
struct MinReducer : Reducer {
    int init() override {
        return numeric_limits<int>::max();
    }

    int add(int a, int b) override {
        return min(a, b);
    }
};

struct MaxReducer : Reducer {
    int init() override {
        return numeric_limits<int>::min();
    }

    int add(int a, int b) override {
        return max(a, b);
    }
};
```

## 多重策略

现在，老板需求改变，他想要 sum 和 product 函数从输入数据直接计算（而不用先读取到一个 vector）！

还好你早已提前抽出公共部分，现在只需要修改 reduce 函数本身就可以了。

SumReducer 和 ProductReducer 无需任何修改，体现了**开闭原则**。

```cpp
int reduce(Reducer *reducer) {
    int res = reducer.init();
    while (true) {
        int tmp;
        cin >> tmp;
        if (tmp == -1) break;
        res = reducer.add(res, tmp);
    }
    return res;
}
```

现在，老板需求又改回来，他突然又想要从 vector 里读取数据了。

在破口大骂老板出尔反尔的同时，你开始思考，这两个函数似乎还是有一些重复可以抽取出来？

```cpp
int cin_reduce(Reducer *reducer) {
    int res = reducer.init();
    while (true) {
        int tmp;
        cin >> tmp;
        if (tmp == -1) break;
        res = reducer.add(res, tmp);
    }
    return res;
}

int vector_reduce(vector<int> v, Reducer *reducer) {
    int res = reducer.init();
    for (int i = 0; i < v.size(); i++) {
        res = reducer.add(res, v[i]);
    }
    return res;
}
```

现在我们只有表示如何计算的类 Reducer 做参数。

你决定，再定义一个表示如何读取的虚类 Inputer。

```cpp
struct Inputer {
    virtual optional<int> fetch() = 0;
};

int reduce(Inputer *inputer, Reducer *reducer) {
    int res = reducer.init();
    while (int tmp = inputer.fetch()) {
        res = reducer.add(res, tmp);
    }
    return res;
}
```

这样，我们满足了**单一职责原则**：每个类只负责一件事。

这里的 Inputer 实际上运用了**迭代器模式**：提供一个抽象接口来**顺序访问**一个集合中各个元素，而又无须暴露该集合的内部表示。

> 底层是 cin 还是 vector？我不在乎！我只知道他可以依次顺序取出数据。

```cpp
struct CinInputer : Inputer {
    optional<int> fetch() override {
        int tmp;
        cin >> tmp;
        if (tmp == -1)
            return nullopt;
        return tmp;
    }
};

struct VectorInputer : Inputer {
    vector<int> v;
    int pos = 0;

    VectorInputer(vector<int> v) : v(v) {}
    
    optional<int> fetch() override {
        if (pos == v.size())
            return nullopt;
        return v[pos++];
    }
};
```

```cpp
reduce(new CinInputer(), new SumReducer());
reduce(new VectorInputer(v), new SumReducer());
reduce(new CinInputer(), new ProductReducer());
reduce(new VectorInputer(v), new ProductReducer());
```

Inputer 负责告诉 reduce 函数如何读取数据，Reducer 负责告诉 reduce 函数如何计算数据。

这就是**依赖倒置原则**：高层模块（reduce 函数）不要直接依赖于低层模块，二者都依赖于抽象（Inputer 和 Reducer 类）来沟通。

## 不要什么东西都塞一块

有些糟糕的实现会把分明不属于同一层次的东西强行放在一起，比如没能分清 Inputer 和 Reducer 类，错误地把他们设计成了一个类！

```cpp
int reduce(Reducer *reducer) {
    int res = reducer.init();
    while (int tmp = reducer.fetch()) {  // fetch 凭什么和 init、add 放在一起？
        res = reducer.add(res, tmp);
    }
    return res;
}
```

fetch 明明属于 IO 操作！但他被错误地放在了本应只负责计算的 Reducer 里！

这导致你必须实现四个类，罗列所有的排列组合：

```cpp
struct CinSumReducer : Reducer { ... };
struct VectorSumReducer : Reducer { ... };
struct CinProductReducer : Reducer { ... };
struct VectorProductReducer : Reducer { ... };
```

这显然是不符合**单一责任原则**的。

满足**单一责任原则**、**开闭原则**、**依赖倒置原则**的代码更加灵活、易于扩展、易于维护。请务必记住并落实起来！
否则即你装模作样地用了虚函数，也一样会导致代码重复、难以维护！

> 老板克扣工资时就不用遵守这些原则

# 适配器模式

刚才的例子中我们用到了 Inputer 虚接口类。

```cpp
struct CinInputer : Inputer {
    optional<int> fetch() override {
        int tmp;
        cin >> tmp;
        if (tmp == -1)
            return nullopt;
        return tmp;
    }
};

struct VectorInputer : Inputer {
    vector<int> v;
    int pos = 0;

    VectorInputer(vector<int> v) : v(v) {}
    
    optional<int> fetch() override {
        if (pos == v.size())
            return nullopt;
        return v[pos++];
    }
};
```

如果我们想要实现：读取到 0 截止，而不是 -1 呢？难道还得给 CinInputer 加个参数？
但是 vector 有时候也可能有读到 -1 就提前截断的需求呀？

这明显违背了**单一责任原则**。

更好的设计是，让 CinInputer 无限读取，永远成功。
然后另外弄一个 StopInputerAdapter，其接受一个 CinInputer 作为构造参数。
当 StopInputerAdapter 被读取时，他会检查是否为 -1，如果已经得到 -1，那么就返回 nullopt，不会进一步调用 CinInputer 了。

StopInputerAdapter 负责处理截断问题，CinInputer 只是负责读取 cin 输入。满足了**单一责任原则**。

```cpp
struct StopInputerAdapter : Inputer {
    Inputer *inputer;
    int stopMark;

    StopInputerAdapter(Inputer *inputer, int stopMark)
        : inputer(inputer)
        , stopMark(stopMark)
    {}

    optional<int> fetch() override {
        auto tmp = inputer.fetch();
        if (tmp == stopMark)
            return nullopt;
        return tmp;
    }
};
```

这里的 StopInputerAdapter 就是一个适配器，他把 CinInputer 的接口（无限读取）叠加上了一个额外功能，读到指定的 stopMark 值就停止，产生了一个新的 Inputer。

```cpp
reduce(new StopInputerAdapter(new CinInputer(), -1), new SumReducer());      // 从 cin 读到 -1 为止
reduce(new StopInputerAdapter(new VectorInputer(v), -1), new SumReducer());  // 从 vector 读到 -1 为止
reduce(new VectorInputer(), new SumReducer());  // 从 vector 读，但无需截断
```

这就是**适配器模式**：将一个类的接口添油加醋，转换成客户希望的另一个接口。

- StopInputerAdapter 这个适配器本身也是一个 Inputer，可以直接作为 reduce 的参数，适应了现有的**策略模式**。
- StopInputerAdapter 并不依赖于参数 Inputer 的底层实现，可以是 CinInputer、也可以是 VectorInputer，满足了**依赖倒置原则**。
- 未来即使新增了不同类型的 Inputer，甚至是其他 InputerAdapter，一样可以配合 StopInputerAdapter 一起使用而无需任何修改，满足了**开闭原则**。

---

如果我们还想实现，过滤出所有正数和零，负数直接丢弃呢？

```cpp
struct FilterInputerAdapter {
    Inputer *inputer;

    FilterInputerAdapter(Inputer *inputer)
        : inputer(inputer)
    {}

    optional<int> fetch() override {
        while (true) {
            auto tmp = inputer.fetch();
            if (!tmp.has_value()) {
                return nullopt;
            }
            if (tmp >= 0) {
                return tmp;
            }
        }
    }
};
```

改进：Filter 的条件不应为写死的 `tmp >= 0`，而应该是传入一个 FilterStrategy，允许用户扩展。

```cpp
struct FilterStrategy {
    virtual bool shouldDrop(int value) = 0;  // 返回 true 表示该值应该被丢弃
};

struct FilterStrategyAbove : FilterStrategy { // 大于一定值（threshold）才能通过
    int threshold;

    FilterStrategyAbove(int threshold) : threshold(threshold) {}

    bool shouldPass(int value) override {
        return value > threshold;
    }
};

struct FilterStrategyBelow : FilterStrategy { // 小于一定值（threshold）才能通过
    int threshold;

    FilterStrategyBelow(int threshold) : threshold(threshold) {}

    bool shouldPass(int value) override {
        return value < threshold;
    }
};

struct FilterInputerAdapter : Inputer {
    Inputer *inputer;
    FilterStrategy *strategy;

    FilterInputerAdapter(Inputer *inputer, FilterStrategy *strategy)
        : inputer(inputer), strategy(strategy)
    {}

    optional<int> fetch() override {
        while (true) {
            auto tmp = inputer.fetch();
            if (!tmp.has_value()) {
                return nullopt;
            }
            if (strategy->shouldPass(tmp)) {
                return tmp;
            }
        }
    }
};
```

FilterStrategy 又可以进一步运用适配器模式：例如我们可以把 FilterStrategyAbove(0) 和 FilterStrategyBelow(100) 组合起来，实现过滤出 0～100 范围内的整数。

```cpp
struct FilterStrategyAnd : FilterStrategy {  // 要求 a 和 b 两个过滤策略都为 true，才能通过
    FilterStrategy *a;
    FilterStrategy *b;

    FilterStrategyAnd(FilterStrategy *a, FilterStrategy *b)
        : a(a), b(b)
    {}

    bool shouldPass(int value) override {
        return a->shouldPass(value) && b->shouldPass(value);
    }
};
```

```cpp
reduce(
    new FilterInputerAdapter(
        new StopInputerAdapter(
            new CinInputer(),
            -1
        ),
        new FilterStrategyAnd(
            new FilterStrategyAbove(0),
            new FilterStrategyBelow(100)
        )
    ),
    new SumReducer());
```

是不是逻辑非常清晰，而且容易扩展呢？

> 实际上函数式和模板元编程更擅长做这种工作，但今天先介绍完原汁原味的 Java 风格面向对象，他们复用代码的思路是共通的。
> 你先学会走路，明天我们再来学习跑步，好吧？

## 跨接口的适配器

适配器模式还可以使原本由于接口不兼容而不能一起工作的那些类可以一起工作，例如一个第三方库提供了类似于我们 Inputer 的输入流接口，也是基于虚函数的。但是他的接口显然不能直接传入我们的 reduce 函数，我们的 reduce 函数只接受我们自己的 Inputer 接口。这时就可以用适配器，把接口翻译成我们的 reducer 能够理解的。

以下是一个自称 “Poost” 的第三方库提供的接口：

```cpp
struct PoostInputer {
    virtual bool hasNext() = 0;
    virtual int getNext() = 0;
};
```

他们要求的用法是先判断 hasNext()，然后才能调用 getNext 读取出真正的值。小彭老师设计了一个 Poost 适配器，把 PoostInputer 翻译成我们的 Inputer：

```cpp
struct PoostInputerAdapter {
    PoostInputer *poostIn;
    optional<int> next;

    PoostInputerAdapter(PoostInputer *poostIn)
        : poostIn(poostIn)
    {}

    optional<int> fetch() override {
        if (next.has_value()) {
            auto res = next;
            next = nullopt;
            return res;
        }

        if (poostIn.hasNext()) {
            return poostIn.getNext();
        } else {
            return nullopt;
        }
    }
};
```

当我们得到一个 PoostInputer 时，如果想要调用我们自己的 reducer，就可以用这个 PoostInputerAdapter 套一层：

```cpp
auto poostStdIn = poost::getStandardInput();
reduce(new PoostInputerAdapter(poostStdIn), new SumReducer());
```

这样就可以无缝地把 PoostInputer 作为 reduce 的参数了。

# 工厂模式

现在你是一个游戏开发者，你的玩家可以装备武器，不同的武器可以发出不同的子弹！

你使用小彭老师教的**策略模式**，把不同的子弹类型作为不同的策略传入 player 函数，造成不同类型的伤害。

```cpp
struct Bullet {
    virtual void explode() = 0;
};

struct AK47Bullet : Bullet {
    void explode() override {
        puts("物理伤害");
    }
};

struct MagicBullet : Bullet {
    void explode() override {
        puts("魔法伤害");
    }
};

void player(Bullet *bullet) {
    bullet->explode();
}

player(new AK47Bullet());
player(new MagicBullet());
```

但是这样就相当于每个玩家只有一发子弹，听个响就没了…

如何允许玩家源源不断地创造新子弹出来？我们可以把“创建子弹”这一过程抽象出来，放在一个“枪”类里。

```cpp
struct Gun {
    virtual Bullet *shoot() = 0;
};

struct AK47Gun : Gun {
    Bullet *shoot() override {
        return new AK47Bullet();
    }
};

struct MagicGun : Gun {
    Bullet *shoot() override {
        return new MagicBullet();
    }
};

void player(Gun *gun) {
    for (int i = 0; i < 100; i++) {
        Bullet *bullet = gun->shoot();
        bullet->explode();
    }
}

player(new AK47Gun());
player(new MagicGun());
```

现在，你的玩家可以直接选择不同的枪了！

这就是所谓的**工厂模式**：“枪”就是“子弹”对象的工厂。
传给玩家的是子弹的工厂——枪，而不是子弹本身。
只要调用工厂的 shoot 函数，玩家可以源源不断地创建新子弹出来。
正所谓授人以鱼不如授人以渔，你的玩家不再是被动接受子弹，而是可以自己创造子弹了！

工厂还可以具有一定的参数，例如我们需要模拟 AK47 可能“受潮”，导致产生的子弹威力降低。
就可以给枪加一个 isWet 参数，给子弹加一个 damage 参数，让 AK47 生成子弹的时候，根据 isWet 为子弹构造函数设置不同的 damage。

```cpp
struct AK47Bullet {
    int damage;

    AK47Bullet(int damage) : damage(damage) {}

    void explode() {
        printf("造成 %d 点物理伤害\n", damage);
    }
};

struct AK47Gun : Gun {
    bool isWet;

    AK47Gun(bool isWet) : isWet(isWet) {}

    Bullet *shoot() override {
        if (isWet)
            return new AK47Bullet(5);  // 受潮了，伤害降低为 5
        else
            return new AK47Bullet(10); // 正常情况下伤害为 10
    }
};
```

我们还可以利用模板自动为不同的子弹类型批量定义工厂：

```cpp
template <class B>
struct GunWithBullet : Gun {
    static_assert(is_base_of<Bullet, B>::value, "B 必须是 Bullet 的子类");

    Bullet *shoot() override {
        return new B();
    }
};

void player(Gun *gun) {
    for (int i = 0; i < 100; i++) {
        Bullet *bullet = gun->shoot();
        bullet->explode();
    }
}

player(new GunWithBullet<AK47Bullet>());
player(new GunWithBullet<MagicBullet>());
};
```

这样就不必每次添加新子弹类型时，都得新建一个相应的枪类型了，进一步避免了代码重复。可见模板元编程完全可与传统面向对象强强联手。

## 超级工厂模式

```cpp
Gun *getGun(string name) {
    if (name == "AK47") {
        return new GunWithBullet<AK47Bullet>();
    } else if (name == "Magic") {
        return new GunWithBullet<MagicBullet>();
    } else {
        throw runtime_error("没有这种枪");
    }
}

player(getGun("AK47"));
player(getGun("Magic"));
```

## RAII 自动管理内存

```cpp
template <class B>
struct GunWithBullet : Gun {
    static_assert(is_base_of<Bullet, B>::value, "B 必须是 Bullet 的子类");

    Bullet *shoot() override {
        return new B();
    }
};

void player(Gun *gun) {
    for (int i = 0; i < 100; i++) {
        Bullet *bullet = gun->shoot();
        bullet->explode();
        delete bullet;  // 刚才没有 delete！会产生内存泄漏！
    }
}

player(new GunWithBullet<AK47Bullet>());
player(new GunWithBullet<MagicBullet>());
```

现在的工厂一般都会返回智能指针就没有这个问题。

具体来说就是用 `unique_ptr<T>` 代替 `T *`，用 `make_unique<T>(xxx)` 代替 `new T(xxx)`。

```cpp
template <class B>
struct GunWithBullet : Gun {
    static_assert(is_base_of<Bullet, B>::value, "B 必须是 Bullet 的子类");

    unique_ptr<Bullet> shoot() override {
        return make_unique<B>();
    }
};

void player(Gun *gun) {
    for (int i = 0; i < 100; i++) {
        auto bullet = gun->shoot();
        bullet->explode();
        // unique_ptr 在退出当前 {} 时会自动释放，不用你惦记着了
    }
}

player(make_unique<GunWithBullet<AK47Bullet>>().get());
player(make_unique<GunWithBullet<MagicBullet>>().get());
```

> 这里 C++ 标准保证了 unique_ptr 的生命周期是这一整行（; 结束前），整个 player 执行期间都活着，不会提前释放
> 正如 `func(string().c_str())` 不会有任何问题，string 要到 func 返回后才释放呢！

只要把所有 `make_unique<T>` 看作 `new T`，把所有的 `unique_ptr<T>` 看作 `T *`，用法几乎一样，但没有内存泄漏，无需手动 delete。

## 工厂模式实战

回到数组求和问题。

```cpp
int sum(vector<int> v) {
    int res = 0;
    for (int i = 0; i < v.size(); i++) {
        res = res + v[i];
    }
    return res;
}

int product(vector<int> v) {
    int res = 1;
    for (int i = 0; i < v.size(); i++) {
        res = res * v[i];
    }
    return res;
}

int average(vector<int> v) {
    int res = 0;
    int count = 0;
    for (int i = 0; i < v.size(); i++) {
        res = res + v[i];
        count = count + 1;
    }
    return res / count;
}
```

我们想要加一个求平均值的函数 average，这该如何与 sum 合起来？

注意因为我们要支持从 CinInputer 读入数据，并不一定像一样 VectorInputer 能够提前得到数组大小，不然也不需要 count 了。

```cpp
int reduce(vector<int> v) {
    int res = ???;              // sum 时这里是 0，product 时这里是 1
    int count? = ???;           // sum 和 product 用不到该变量，只有 average 需要
    for (int i = 0; i < v.size(); i++) {
        res = res ??? v[i];   // sum 时这里是 +，product 时这里是 *
        count? = count? ???;  // average 时这里还需要额外修改 count 变量！
    }
    return res;
}
```

看来我们需要允许 Reducer 的 init() 返回 “任意数量的状态变量”！
以前的设计让 init() 只能返回单个 int 是个错误的决定。
这时候就可以把 “任意数量的状态变量” 封装成一个新的类。
然后改为由这个类负责提供虚函数 add()。
且只需要提供一个右侧参数了，左侧的 res 变量已经存在 ReducerState 体内了。

```cpp
struct ReducerState {
    virtual void add(int val) = 0;
    virtual int result() = 0;
};

struct Reducer {
    virtual unique_ptr<ReducerState> init() = 0;
};

struct SumReducerState : ReducerState {
    int res;

    SumReducerState() : res(0) {}

    void add(int val) override {
        res = res + val;
    }

    int result() override {
        return res;
    }
};

struct ProductReducerState : ReducerState {
    int res;

    ProductReducerState() : res(1) {}

    void add(int val) override {
        res = res * val;
    }

    int result() override {
        return res;
    }
};

struct AverageReducerState : ReducerState {
    int res;
    int count;

    AverageReducerState() : res(0), count(0) {}

    void add(int val) override {
        res = res + val;
        count = count + 1;
    }

    int result() override {
        return res / count;
    }
};

struct SumReducer : Reducer {
    unique_ptr<ReducerState> init() override {
        return make_unique<SumReducerState>();
    }
};

struct ProductReducer : Reducer {
    unique_ptr<ReducerState> init() override {
        return make_unique<ProductReducerState>();
    }
};

struct AverageReducer : Reducer {
    unique_ptr<ReducerState> init() override {
        return make_unique<AverageReducerState>();
    }
};
```

这里 Reducer 就成了 ReducerState 的工厂。

```cpp
int reduce(Inputer *inputer, Reducer *reducer) {
    unique_ptr<ReducerState> state = reducer->init();
    while (auto val = inputer->fetch()) {
        state->add(val);
    }
    return state->result();
}

int main() {
    vector<int> v;
    reduce(make_unique<VectorInputer>(v).get(), make_unique<SumReducer>().get());
    reduce(make_unique<VectorInputer>(v).get(), make_unique<ProductReducer>().get());
    reduce(make_unique<VectorInputer>(v).get(), make_unique<AverageReducer>().get());
}
```

---

现在，老板需求改变，他想要**并行**的 sum 和 product 函数！

并行版需要创建很多个任务，每个任务需要有一个自己的中间结果变量，最后的结果计算又需要一个中间变量。
还好你早已提前采用工厂模式，允许函数体内多次创建 ReducerState 对象。

```cpp
int reduce(Inputer *inputer, Reducer *reducer) {
    tbb::task_group g;
    list<unique_ptr<ReducerState>> local_states;
    vector<int> chunk;
    auto enqueue_chunk = [&]() {
        local_chunks.emplace_back();
        g.run([chunk = move(chunk), &back = local_chunks.back()]() {
            auto local_state = reducer->init();
            for (auto &&c: chunk) {
                local_state->add(c);
            }
            back = move(local_state); // list 保证已经插入元素的引用不会失效，所以可以暂存 back 引用
        });
        chunk.clear();
    };
    while (auto tmp = inputer->fetch()) {
        if (chunk.size() < 64) { // 还没填满 64 个
            chunk.push_back(tmp);
        } else { // 填满了 64 个，可以提交成一个单独任务了
            enqueue_chunk();
        }
    }
    if (chunk.size() > 0) {
        enqueue_chunk(); // 提交不足 64 个的残余项
    }
    g.wait();
    auto final_state = reducer->init();
    for (auto &&local_state: local_states) {
        res = final_state->add(local_state->result());
    }
    return final_state->result();
}
```

只需要把 reducer 参数替换为 MinReducer、AverageReducer……就自动适用于不同的计算任务，而不用为他们每个单独编写并行版本的代码。

课后作业：使用模板批量定义所有的 Reducer！例如：

```cpp
using MinReducer = ReducerWithState<MinReducerState>;
...
```

# 享元模式

在二维游戏开发中，常常会提到一种称为 Sprite（精灵贴图）的黑话，实际上就是每个对象自己有一张贴图，贴图跟着物体的位置走。

```cpp
struct Bullet {
    glm::vec3 position;
    glm::vec3 velocity;
    vector<char> texture;

    void draw() {
        glDrawPixels(position, texture);
    }
};
```

texture 里面存储着贴图的 RGB 数据，他直接就是 Bullet 的成员。
这样的话，如果我们的玩家打出了 100 颗子弹，就需要存储 100 个贴图数组。
如果我们的玩家同时打出了 1000 颗子弹，就需要存储 1000 个贴图数组。
这样的话，内存消耗将会非常大。然而所有同类型的 Bullet，其贴图数组其实是完全相同的，完全没必要各自存那么多份拷贝。

为解决这个问题，我们可以使用**享元模式**：共享多个对象之间**相同**的部分，节省内存开销。

这里每颗子弹的 position、velocity 显然都是各有不同的，不可能所有子弹都在同一个位置上。
但是很多子弹都会有着相同的贴图，只有不同类型的子弹贴图会不一样。
比如火焰弹和寒冰弹会有不同的贴图，但是当场上出现 100 颗火焰弹时，显然不需要拷贝 100 份完全相同的火焰弹贴图。

```cpp
struct Sprite {  // Sprite 才是真正持有（很大的）贴图数据的
    vector<char> texture;

    void draw(glm::vec3 position) {
        glDrawPixels(position, texture);
    }
};

struct Bullet {
    glm::vec3 position;
    glm::vec3 velocity;
    shared_ptr<Sprite> sprite;  // 允许多个子弹对象共享同一个精灵贴图的所有权

    void draw() {
        sprite->draw(position);  // 转发给 Sprite 让他帮忙在我的位置绘制贴图
    }
};
```

需要绘制子弹时，Bullet 的 draw 只是简单地转发给 Sprite 类的 draw。
只要告诉 Sprite 子弹的位置就行，贴图数据已经存在 Sprite 内部，让他来负责真正绘制。
Bullet 类只需要专注于位置、速度的更新即可，不必去操心着贴图绘制的细节，实现了解耦。

这种函数调用的转发也被称为**代理模式**。

## 代理模式

这样还有一个好处那就是，Sprite 可以设计成一个虚函数接口类：

```cpp
struct Sprite {
    virtual void draw(glm::vec3 position) = 0;
};

struct FireSprite {
    vector<char> fireTexture;

    FireSprite() : fireTexture(loadTexture("fire.jpg")) {}

    void draw(glm::vec3 position) override {
        glDrawPixels(position, fireTexture);
    }
};

struct IceSprite { // 假如寒冰弹需要两张贴图，也没问题！因为虚接口类允许子类有不同的成员，不同的结构体大小
    vector<char> iceTexture1;
    vector<char> iceTexture2;

    IceSprite()
    : iceTexture1(loadTexture("ice1.jpg"))
    , iceTexture2(loadTexture("ice2.jpg"))
    {}

    void draw(glm::vec3 position) override {
        glDrawPixels(position, iceTexture1);
        glDrawPixels(position, iceTexture2);
    }
};
```

```cpp
struct Bullet {
    glm::vec3 position;
    glm::vec3 velocity;
    shared_ptr<Sprite> sprite;  // Sprite 负责含有虚函数

    void draw() {  // Bullet 的 draw 就不用是虚函数了！
        sprite->draw(position);
    }
};
```



# 组件模式

```cpp
```


# 虚函数常见问题辨析

## 返回 bool 的虚函数






## 课后作业

你拿到了一个大学生计算器的大作业：

```cpp
int main() {
    char c;
    cout << "请输入第一个数：";
    cin >> a;
    cout << "请输入第二个数：";
    cin >> b;
    cout << "请输入运算符：";
    cin >> c;
    if (c == '+') {
        cout << a + b;
    } else if (c == '-') {
        cout << a - b;
    } else if (c == '*') {
        cout << a * b;
    } else if (c == '/') {
        cout << a / b;
    } else {
        cout << "不支持的运算符";
    }
}
```

你开始用策略模式改造它：

```cpp
struct Calculator {
    virtual int calculate(int a, int b) = 0;
};

struct AddCalculator : Calculator {
    int calculate(int a, int b) override {
        return a + b;
    }
};

struct SubCalculator : Calculator {
    int calculate(int a, int b) override {
        return a - b;
    }
};

struct MulCalculator : Calculator {
    int calculate(int a, int b) override {
        return a * b;
    }
};

struct DivCalculator : Calculator {
    int calculate(int a, int b) override {
        return a / b;
    }
};

Calculator *getCalculator(char c) {
    if (c == '+') {
        calculator = new AddCalculator();
    } else if (c == '-') {
        calculator = new SubCalculator();
    } else if (c == '*') {
        calculator = new MulCalculator();
    } else if (c == '/') {
        calculator = new DivCalculator();
    } else {
        throw runtime_error("不支持的运算符");
    }
};

int main() {
    char c;
    cout << "请输入第一个数：";
    cin >> a;
    cout << "请输入第二个数：";
    cin >> b;
    cout << "请输入运算符：";
    cin >> c;
    Calculator *calculator = getCalculator(c);
    cout << calculator->calculate(a, b);
}
```
