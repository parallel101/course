# 游戏开发中经常用到的设计模式

- 单例模式
- 模板模式
- 状态模式
- 原型模式
- CRTP 模式
- 组件模式
- 观察者模式
- 发布-订阅模式
- 访问者模式

## 单例模式

通常用于游戏中的全局管理类，保证整个程序（进程）中只有一个实例对象存在。有很多种常见的写法：

1. 作为全局变量（饿汗模式）

```cpp
Game game;
```

效果：在程序启动时就会创建 `game` 对象，之后可以直接使用。

2. 作为函数内部的 static 变量（懒汗模式）

```cpp
Game &getGame() {
    static Game game;
    return game;
}

getGame().updatePlayers();
```

效果：第一次调用 `getGame()` 时会初始化，之后的调用会直接返回上次创建的实例。

根据你的需要，如果你需要在程序一启动时 game 对象就可用，就用饿汗模式。

如果 game 的初始化需要某些条件，例如创建 Game 类前需要 OpenGL 初始化，那么可用懒汗模式：

```cpp
int main() {
    glfwInit();                 // 初始化 OpenGL
    getGame().initialize();     // 第一次调用 getGame 会初始化 game 单例
    getGame().updatePlayers();  // 之后的调用总是返回对同一个 game 单例的引用
}
```

> 提示：如果要把单例对象的定义放在头文件中，务必添加 inline 修饰符，而不是 static，否则会导致多个 cpp 文件各自有一个 Game 对象。

```cpp
// Game.hpp

inline Game game;

inline Game &getGame() {
    static Game game;
    return game;
}
```

### 封装在类内部

由于所有单例全部暴露在全局名字空间，容易产生混乱。
一般会把单例对象或函数封装在类内部，并且把 Game 的构造函数设为 private，避免用户不慎直接创建出本应只有单个实例的 Game 类。

1. 作为全局变量（饿汗模式）

```cpp
struct Game {
    ...

    Game(Game &&) = delete;

private:
    Game() { ... }

public:
    inline static Game instance;  // 如果定义在头文件中，需要 inline！
};

Game::instance.updatePlayers();
```

2. 作为函数内部的 static 变量（懒汗模式）

```cpp
struct Game {
    ...

    Game(Game &&) = delete;

private:
    Game() { ... }

public:
    inline static Game &instance() {  // 这里的 inline 可以省略，因为类体内就地实现的函数自带 inline 效果
        static Game game;
        return game;
    }
};

Game::instance().updatePlayers();
```

### 通用的单例模式模板

```cpp
template <class T>
inline T &singleton() {  // 这里的 inline 可以省略，因为就地实现的模板函数自带 inline 效果
    // 只有第一次进入时会构造一遍 T，之后不会再构造
    // 不同的 T 会实例化出不同的 singleton 实例，各自体内的 static 变量独立计算，互不干扰
    static T inst;
    return inst;
}

singleton<Game>().updatePlayers();
singleton<Other>().someMethod();
```

任何类型 T，只要以 `singleton<T>()` 形式获取，都能保证每个 T 都只有一份对象。（前提是你不要再 `T()` 创建对象）

## 模板模式

> 注意：模板模式和 C++ 的模板并没有必然关系！模板模式只是一种思想，可以用模板实现，也可以用虚函数实现（大多反而是用虚函数实现的）

模板模式用于封装游戏中一些相似的处理逻辑，把共同的部分集中到一个基类，把不同的细节部分留给子类实现。

和策略模式很像，只不过这里接收策略的直接就是基类自己。

例如，一个角色固定每一帧需要移动 3 次，然后绘制 1 次。显然需要把“移动”和“绘制”作为两个虚函数接口，让子类来实现。

```cpp
struct Character {
    virtual void draw() = 0;
    virtual void move() = 0;
};

struct Player : Character {
    void draw() override {
        drawPlayer();
    }

    void move() override {
        movePlayer();
    }
};

struct Enemy : Character {
    void draw() override {
        drawEnemy();
    }

    void move() override {
        moveEnemy();
    }
};
```

如果让负责调用 Character 的人来实现每一帧需要移动 3 次 + 绘制 1 次的话，就破坏了开闭原则。

```cpp
struct Game {
    vector<Character *> chars;

    void update() {
        for (auto &&c: chars) {
            c->move();
            c->move();
            c->move();
            c->draw();
        }
    }
}
```

改为把移动 3 次 + 绘制 1 次封装为一个 Character 的普通函数 update。

```cpp
struct Character {
protected:
    virtual void draw() = 0;
    virtual void move() = 0;

public:
    void update() {
        move();
        move();
        move();
        draw();
    }
};

struct Game {
    vector<Character *> chars;

    void update() {
        for (auto &&c: chars) {
            c->update();
        }
    }
}
```

这样调用者就很轻松了，不必关心底层细节，而 update 也只通过接口和子类通信，满足开闭原则和依赖倒置原则。

### 模板模式还是策略模式：如何选择？

当一个对象涉及很多策略时，用策略模式；当只需要一个策略，且需要用到基类的成员时，用模板模式。

例如，一个角色的策略有移动策略和攻击策略，移动方式有“走路”、“跑步”两种，攻击策略又有“平A”、“暴击”两种。

那么就用策略模式，让角色分别指向移动策略和攻击策略的指针。

```cpp
struct Character {
    MoveStrategy *moveStrategy;
    AttackStrategy *attackStrategy;

    void update() {
        if (isKeyPressed(GLFW_KEY_S) {
            moveStrategy->move();
        } else if (isKeyPressed(GLFW_KEY_W)) {
            moveStrategy->run();
        }
        while (auto enemy = Game::instance().findEnemy(range)) {
            attackStrategy->attack(enemy);
        }
    }
};
```

而如果只有一个策略，比如武器类，只需要攻击策略，并且攻击策略需要知道武器的伤害值、射程、附魔属性等信息，那就适合模板模式。

```cpp
struct Weapon {
protected:
    double damage;
    double charge;
    MagicFlag magicFlags;
    double range;

    virtual void attack(Enemy *enemy);

public:
    void update() {
        while (auto enemy = Game::instance().findEnemy(range)) {
            attack(enemy);
        }
    }
};
```

### 最常见的是 do_xxx 封装

例如，一个处理字符串的虚接口类：

```cpp
struct Converter {
    virtual void process(const char *s, size_t len) = 0;
};
```

这个接口是考虑 **实现 Converter 子类的方便**，对于 **调用 Converter 的用户** 使用起来可能并不方便。

这时候就可以运用模板模式，把原来的虚函数接口改为 protected 的函数，且名字改为 do_process。

```cpp
struct Converter {
protected:
    virtual void do_process(const char *s, size_t len) = 0;

public:
    void process(string_view str) {
        return do_process(str.data(), str.size());
    }

    void process(string str) {
        return do_process(str.data(), str.size());
    }

    void process(const char *cstr) {
        return do_process(cstr, strlen(cstr));
    }
};
```

实现 Converter 的子类时，重写他的 `do_process` 函数，这些函数是 protected 的，只能被继承了 Converter 的子类访问和重写。

外层用户只能通过 Converter 基类封装好的 `process` 函数，避免外层用户直接干涉底层细节。

标准库中的 `std::pmr::memory_resource`、`std::codecvt` 等都运用了 do_xxx 式的模板模式封装。

## 状态模式

游戏中的角色通常有多种状态，例如，一个怪物可能有“待机”、“巡逻”、“追击”、“攻击”等多种状态，而每种状态下的行为都不一样。

如果用一个枚举变量来表示当前状态，那每次就都需要用 switch 来处理不同的状态。

```cpp
enum MonsterState {
    Idle,
    Chase,
    Attack,
};

struct Monster {
    MonsterState state = Idle;

    void update() {
        switch (state) {
            case Idle:
                if (seesPlayer())
                    state = Chase;
                break;
            case Chase:
                if (canAttack())
                    state = Attack;
                else if (!seesPlayer())
                    state = Idle;
                break;
            case Attack:
                if (!seesPlayer())
                    state = Idle;
                break;
        }
    }
};
```

这或许性能上有一定优势，缺点是，所有不同状态的处理逻辑堆积在同一个函数中，如果有多个函数（不只是 update），那么每添加一个新状态就需要修改所有函数，不符合开闭原则。

而且如果不同的状态含有不同的额外数值需要存储，比如 Chase 状态需要存储当前速度，那就需要在 Monster 类中添加 speed 成员，而 state 不为 Chase 时又用不到这个成员，非常容易扰乱思维。

### 状态不是枚举，而是类

为此，提出了状态模式，将不同状态的处理逻辑分离到不同的类中。他把每种状态抽象为一个类，状态是一个对象，让角色持有表示当前状态的对象，用状态对象的虚函数来表示处理逻辑，而不必每次都通过 if 判断来执行不同的行为。

```cpp
struct Monster;

struct State {
    virtual void update(Monster *monster) = 0;
};

struct Idle : State {
    void update(Monster *monster) override {
        if (monster->seesPlayer()) {
            monster->setState(new Chase());
        }
    }
};

struct Chase : State {
    void update(Monster *monster) override {
        if (monster->canAttack()) {
            monster->setState(new Attack());
        } else if (!monster->seesPlayer()) {
            monster->setState(new Idle());
        }
    }
};

struct Attack : State {
    void update(Monster *monster) override {
        if (!monster->seesPlayer()) {
            monster->setState(new Idle());
        }
    }
};

struct Monster {
    State *state = new Idle();

    void update() {
        state->update(this);
    }

    void setState(State *newState) {
        delete state;
        state = newState;
    }
};
```

## 原型模式

原型模式用于复制现有的对象，且新对象的**属性**和**类型**与原来相同。如何实现？

1. 为什么拷贝构造函数不行？

拷贝构造函数只能用于类型确定的情况，对于具有虚函数，可能具有额外成员的多态类型，会发生 object-slicing，导致拷贝出来的类型只是基类的部分，而不是完整的子类对象。

```cpp
RedBall ball;
Ball newball = ball;  // 错误：发生了 object-slicing！现在 newball 的类型只是 Ball 了，丢失了 RedBall 的信息
```

2. 为什么拷贝指针不行？

指针的拷贝是浅拷贝，而我们需要的是深拷贝。

```cpp
Ball *ball = new RedBall();
Ball *newball = ball;  // 错误：指针的拷贝是浅拷贝！newball 和 ball 指向的仍然是同一对象
```

3. 需要调用到真正的构造函数，同时又基于指针

```cpp
Ball *ball = new RedBall();
Ball *newball = new RedBall(*dynamic_cast<RedBall *>(ball));  // 可以，但是这里显式写出了 ball 内部的真正类型，违背了开闭原则
```

4. 将拷贝构造函数封装为虚函数

原型模式将对象的拷贝方法作为虚函数，返回一个虚接口的指针，避免了直接拷贝类型。但虚函数内部会调用子类真正的构造函数，实现深拷贝。

对于熟悉工厂模式的同学：原型模式相当于把每个对象变成了自己的工厂，只需要有一个现有的对象，就能不断复制出和他相同类型的对象来。

```cpp
struct Ball {
    virtual Ball *clone() = 0;
};

struct RedBall : Ball {
    Ball *clone() override {
        return new RedBall(*this);  // 调用 RedBall 的拷贝构造函数
    }
};

struct BlueBall : Ball {
    Ball *clone() override {
        return new BlueBall(*this);  // 调用 BlueBall 的拷贝构造函数
    }

    int someData;  // 如果有成员变量，也会一并被拷贝到
};
```

好处是，调用者无需知道具体类型，只需要他是 Ball 的子类，就可以克隆出一份完全一样的子类对象来，且返回的也是指针，不会发生 object-slicing。

```cpp
Ball *ball = new RedBall();
...
Ball *newball = ball->clone();  // newball 的类型仍然是 RedBall
```

### clone 返回为智能指针

```cpp
struct Ball {
    virtual unique_ptr<Ball> clone() = 0;
};

struct RedBall : Ball {
    unique_ptr<Ball> clone() override {
        return make_unique<RedBall>(*this);  // 调用 RedBall 的拷贝构造函数
    }
};

struct BlueBall : Ball {
    unique_ptr<Ball> clone() override {
        return make_unique<BlueBall>(*this);  // 调用 BlueBall 的拷贝构造函数
    }

    int someData;  // 如果有成员变量，也会一并被拷贝到新对象中
};
```

这样就保证了内存不会泄漏。

> 如果调用者需要的是 shared_ptr，怎么办？

答：unique_ptr 可以隐式转换为 shared_ptr。

> 如果调用者需要的是手动 delete 的原始指针，怎么办？

答：unique_ptr 可以通过 release，故意造成一次内存泄漏，成为需要手动管理的原始指针。

### CRTP 模式自动实现 clone

CRTP（Curiously Recurring Template Pattern）是一种模板元编程技术，它可以在编译期间把派生类的类型作为模板参数传递给基类，从而实现一些自动化的功能。

特点是，继承一个 CRTP 类时，需要把子类本身作为基类的模板参数。

> 并不会出现循环引用是因为，用到子类的具体类型是在基类的成员函数内部，而不是直接在基类内部，而模板类型的成员函数的实例化是惰性的，用到了才会实例化。

```cpp
template <class Derived>
struct Pet {
    void feed() {
        Derived *that = static_cast<Derived *>(this);
        that->speak();
        that->speak();
    }
};

struct CatPet : Pet<CatPet> {
    void speak() {
        puts("Meow!");
    }
};

struct DogPet : Pet<DogPet> {
    void speak() {
        puts("Bark!");
    }
};
```

一般的象牙塔理论家教材中都会告诉你，CRTP 是用于取代虚函数，更高效地实现模板模式，好像 CRTP 就和虚函数势不两立。

但小彭老师的编程实践中，CRTP 常常是和虚函数一起出现的好搭档。

例如 CRTP 可以帮助原型模式实现自动化定义 clone 虚函数，稍后介绍的访问者模式中也会用到 CRTP。

```cpp
struct Ball {
    virtual unique_ptr<Ball> clone() = 0;
};

template <class Derived>
struct BallImpl : Ball {  // 自动实现 clone 的辅助工具类
    unique_ptr<Ball> clone() override {
        Derived *that = static_cast<Derived *>(this);
        return make_unique<Derived>(*that);
    }
};

struct RedBall : BallImpl<RedBall> {
    // unique_ptr<Ball> clone() override {       // BallImpl 自动实现的 clone 等价于
    //     return make_unique<RedBall>(*this);  // 调用 RedBall 的拷贝构造函数
    // }
};

struct BlueBall : BallImpl<BlueBall> {
    // unique_ptr<Ball> clone() override {       // BallImpl 自动实现的 clone 等价于
    //     return make_unique<BlueBall>(*this);  // 调用 BlueBall 的拷贝构造函数
    // }
};
```

> 在小彭老师自主研发的 Zeno 中，对象类型 `zeno::IObject` 的深拷贝就运用了 CRTP 加持的原型模式。

## 组件模式

游戏中的物体（游戏对象）通常由多个组件组成，例如，一个角色可能由“角色控制器”、“角色外观”、“角色动画”等组件组成，一个子弹可能由“子弹物理”、“子弹外观”等组件组成。

组件模式是**游戏开发领域最重要的设计模式**，它将游戏对象分为多个组件，每个组件只关心自己的逻辑，而不关心其他组件的逻辑。

蹩脚的游戏开发者（通常是 985 量产出来的象牙塔巨婴）会把每个组件写成一个类，然后使用“多重继承”继承出一个玩家类来，并恬不知耻地声称“我也会组件模式了”。

然而，这样的缺点有：

1. 游戏开发中普遍涉及到 update 函数，而玩家类的 update 需要轮流调用每个组件的 update 函数。

而多重继承一旦遇到重名的 update 函数，会直接报错 “有歧义的函数名” 摆烂不干了，需要你手写新的 update 函数。

```cpp
struct Player : PlayerController, PlayerAppearance, PlayerAnimation {
    void update() {
        PlayerController::update();
        PlayerAppearance::update();
        PlayerAnimation::update();
    }
};
```

2. C++（和大多数非脚本语言都）不支持运行时添加或删除基类，也就是说，如果要添加一个新角色，或是修改现有角色的逻辑，就需要重新编译一遍整个游戏的源码。

在网络游戏中，更新 DLL 和更新资产（图片、音频、模型等）是完全不同的。

- 对于服务端而言，更新 DLL 需要停机更新，更新资产不需要，DLL 可以被编程允许动态加载新的贴图。
- 对于客户端而言，更新 DLL 需要重新走一遍很长的 App 审核流程（因为直接运行于手机上的 C++ 可以轻松植入病毒），而更新资产的审核流程短得多，甚至干脆无需审核。

因此，游戏开发者很少会把游戏逻辑直接写死在 C++ 中，这会让更新游戏逻辑（例如修复 BUG）需要停机更新。（例如明日方舟每次停机更新都会给玩家发 200 合成玉）

> 你经常看到游戏领域的 “C++ 开发岗” 实际上是 “解释器开发”。

游戏开发者会把经常需要维护和更新的游戏逻辑写在如 Lua、Python 等脚本语言中，然后在 C++ 中集成一个 Lua、Python 解释器，根据解释器的调用结果，动态创建出 C++ 对象，然后把这些 C++ 对象当作组件添加到游戏对象上。

当出现 BUG 时，只需要修改这些脚本语言的代码，然后以“资产”的形式，快速走一遍审核流程，就可以修复 BUG，无需停机更新。（例如明日方舟有时候会“资源已过期”“正在下载资源”，有时是更新了图片资源，也可能是在脚本语言里动态修复了 BUG）

3. Java 和 C# 都没有多重继承。你让人家基于 C# 的 Unity 怎么活？

因此，真正的组件模式都会允许动态插入组件，而不是编译期写死。除非你是某些象牙塔的一次性沙雕大作业。

游戏对象组件化后，可以灵活地组合出不同的游戏对象，而不必为每一种组合都写一个类。

```cpp
struct Component {
    virtual void update(GameObject *go) = 0;
    virtual ~Component() = default;  // 注意！
};

struct GameObject {
    vector<Component *> components;

    void add(Component *component) {
        components.push_back(component);
    }

    void update() {
        for (auto &&c: components) {
            c->update(this);
        }
    }
};
```

注意：Component 的析构函数必须为虚函数。否则，当 Component 被 delete 时，只会调用到 Component 这个基类的析构函数，而不会调用到子类的析构函数。

否则，如果你的子类有 string、vector 这种持有内存资源的容器类，会发生内存泄漏，导致游戏运行越久内存占用越大。

> 神奇的是，如果你的 Component 全部都是用 make_shared 创建的，那就没有内存泄漏了，这得益于 shared_ptr 会对 deleter 做类型擦除。
> make_unique 和 new 创建的就会泄漏，因为他们 delete 时是以基类指针去 delete 的，而 shared_ptr 会在构造时就记住子类的 deleter。

所有组件，都支持 update（每帧更新）操作：

```cpp
struct Movable : Component {
    glm::vec3 position;
    glm::vec3 velocity;

    void update(GameObject *go) override {
        position += velocity * dt;
    }
};
```

```cpp
struct LivingBeing : Component {
    int ageLeft;

    void update(GameObject *go) override {
        if (ageLeft < 0)
            go->kill();
        else
            ageLeft -= 1;
    }
};
```

### 组件的创建

组件有两种创建方式：

1. 组件作为一个普通对象，由 GameObject 的构造函数创建。

```cpp
struct Player : GameObject {
    Movable *movable;
    LivingBeing *livingBeing;
    PlayerController *playerController;
    PlayerAppearance *playerAppearance;

    Player() {
        movable = new Movable();
        livingBeing = new LivingBeing(42);
        playerController = new PlayerController();
        playerAppearance = new PlayerAppearance();
        add(movable);
        add(livingBeing);
        add(playerController);
        add(playerAppearance);
    }
};
```

2. 不再需要定义 Player 类及其构造函数了，只需一个普通函数创建具有 Player 所需所有组件的 GameObject 对象即可。

```cpp
GameObject *makePlayer() {
    GameObject *go = new GameObject();

    go->add(new Movable());
    go->add(new LivingBeing(42));
    go->add(new PlayerController());
    go->add(new PlayerAppearance());

    return go;
}
```

正经游戏引擎都采用后者，不用添加 C++ 源码，只是从 xml 等配置文件读取每个类所依赖的组件，就能创建新的玩家类，方便动态更新游戏逻辑而无需重新发布 dll。

### 组件之间如何通信

缺点是，组件之间的通信需要通过 GameObject 来实现，而 GameObject 并不知道它的组件是什么，这样就无法直接访问组件的成员。

例如，PlayerController 组件想要改变 Movable 组件的 velocity，就无法直接改。

```cpp
struct PlayerController : Component {
    void update(GameObject *go) override {
        if (isKeyPressed(GLFW_KEY_W)) {
            go->velocity.y += 1; // 错误！velocity 是 Movable 组件的成员，而不是 GameObject 里直接有的
        }
        if (isKeyPressed(GLFW_KEY_S)) {
            go->velocity.y -= 1;
        }
        if (isKeyPressed(GLFW_KEY_A)) {
            go->velocity.x -= 1;
        }
        if (isKeyPressed(GLFW_KEY_D)) {
            go->velocity.x += 1;
        }
    }
};
```

如何解决组件之间通信难的问题？

1. 把常用的字段，例如 position 和 velocity 直接放在 GameObject 里，供所有组件直接访问。

```cpp
struct GameObject {
    glm::vec3 position;
    glm::vec3 velocity;

    ...
};
```

2. 允许用户根据其他组件的类型，直接获取出其他组件的指针，即可访问其成员。

```cpp
struct PlayerController : Component {
    void update(GameObject *go) override {
        Movable *movable = go->getComponent<Movable>();
        if (!movable) {
            throw runtime_error("这个对象似乎不支持移动");
        }
        if (isKeyPressed(GLFW_KEY_W)) {
            movable->velocity.y += 1;
        }
        if (isKeyPressed(GLFW_KEY_S)) {
            movable->velocity.y -= 1;
        }
        if (isKeyPressed(GLFW_KEY_A)) {
            movable->velocity.x -= 1;
        }
        if (isKeyPressed(GLFW_KEY_D)) {
            movable->velocity.x += 1;
        }
    }
};
```

然而，getComponent 如何实现？

```cpp
struct GameObject {
    template <class T>
    T *getComponent() {
        for (auto &&c: components) {
            if (T *t = dynamic_cast<T *>(c)) {
                return t;
            }
        }
        return nullptr;
    }
};
```

用到了 `dynamic_cast`，这是比较低效的一种实现方式，而且也不符合开闭原则。

更好的实现方式是利用 typeid 做 map 的键，加速查找。没有性能问题，但依然不符合开闭原则。

```cpp
struct GameObject {
    unordered_map<type_index, Component *> components;

    template <class T>
    T *getComponent() {
        if (auto it = components.find(typeid(T)); it != components.end()) {
            return dynamic_cast<T *>(it->second);
        } else {
            return nullptr;
        }
    }

    void add(Component *component) {
        components[typeid(*component)] = component;
    }
};
```

3. 让 PlayerController 发出指定类型的消息对象，由 Movable 检查并处理。

消息类型也是多态的，初学者可以先通过 `dynamic_cast` 实现类型检查。稍后我们会介绍更专业的访问者模式。

通常来说，我们只能把子类指针转换为基类指针。

而 dynamic_cast 可以把基类指针转换为子类指针。

如果他指向的对象确实就是那个子类类型的话，就正常返回子类指针了。

否则，如果类型不匹配，`dynamic_cast` 会返回 nullptr。只需判断返回的指针是不是 nullptr 就知道是否类型匹配了。

### 观察者模式

```cpp
struct Message {
    virtual ~Message() = default;  // C++ 规定：只有多态类型才能 dynamic_cast，这里我们用不到虚函数，那就只让析构函数为虚函数，即可使 Message 变为多态类型
};

struct MoveMessage : Message {
    glm::vec3 velocityChange;
};

struct Component {
    virtual void update(GameObject *go) = 0;
    virtual void handleMessage(Message *msg) = 0;
    virtual ~Component() = default;
};

struct Movable : Component {
    glm::vec3 position;
    glm::vec3 velocity;

    void handleMessage(Message *msg) override {
        // 所有不同的消息类型都会进入此函数
        if (MoveMessage *mm = dynamic_cast<MoveMessage *>(msg)) {
            // 但只有真正类型为 MoveMessage 的消息会被处理
            velocity += mm->velocityChange;
        }
    }
};

struct PlayerController : Component {
    void update(GameObject *go) override {
        if (isKeyPressed(GLFW_KEY_W)) {
            MoveMessage mm;
            mm.velocityChange.y += 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_S)) {
            MoveMessage mm;
            mm.velocityChange.y -= 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_A)) {
            MoveMessage mm;
            mm.velocityChange.x -= 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_D)) {
            MoveMessage mm;
            mm.velocityChange.x += 1;
            go->send(&mm);
        }
    }
};
```

```cpp
struct GameObject {
    vector<Component *> components;

    void add(Component *component) {
        components.push_back(component);
    }

    void update() {
        for (auto &&c: components) {
            c->update(this);
        }
    }

    void send(Message *msg) {
        for (auto &&c: components) {
            c->handleMessage(msg);
        }
    }
};
```

这就是所谓的观察者模式，由于每个组件都可以收到所有消息，因此，可以实现组件之间的通信。

但这样做的缺点是，每个组件都需要处理所有消息，不论是否是自己需要的，如果组件数量多，消息类型又多，就会出现性能问题。

### 发布-订阅模式

发布-订阅模式是观察者模式的升级版，由一个中心的事件总线来管理消息的分发。事件总线通常作为 GameObject 的成员出现。

每个组件可以订阅自己感兴趣的消息类型，当事件总线收到消息时，只把消息分发给订阅者，而不是所有组件。

```cpp
struct GameObject {
    vector<Component *> components;
    unordered_map<type_index, vector<Component *>> subscribers;  // 事件总线

    template <class EventType>
    void subscribe(Component *component) {
        subscribers[type_index(typeid(EventType))].push_back(component);
    }

    template <class EventType>
    void send(EventType *msg) {
        for (auto &&c: subscribers[type_index(typeid(EventType))]) {
            c->handleMessage(msg);
        }
    }

    void add(Component *component) {
        components.push_back(component);
        component->subscribeMessages(this);
    }

    void update() {
        for (auto &&c: components) {
            c->update(this);
        }
    }
};

struct Component {
    virtual void update(GameObject *go) = 0;
    virtual void subscribeMessages(GameObject *go) = 0;
    virtual void handleMessage(Message *msg) = 0;
    virtual ~Component() = default;
};

struct Movable : Component {
    glm::vec3 position;
    glm::vec3 velocity;

    void subscribeMessages(GameObject *go) {
        go->subscribe<MoveMessage>(this);
    }

    void handleMessage(Message *msg) override {
        if (MoveMessage *mm = dynamic_cast<MoveMessage *>(msg)) {
            velocity += mm->velocityChange;
        }
    }
};

struct PlayerController : Component {
    void update(GameObject *go) override {
        if (isKeyPressed(GLFW_KEY_W)) {
            MoveMessage mm;
            mm.velocityChange.y += 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_S)) {
            MoveMessage mm;
            mm.velocityChange.y -= 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_A)) {
            MoveMessage mm;
            mm.velocityChange.x -= 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_D)) {
            MoveMessage mm;
            mm.velocityChange.x += 1;
            go->send(&mm);
        }
        if (isKeyPressed(GLFW_KEY_SPACE)) {
            JumpMessage jm;
            go->send(&jm);
        }
    }
};
```

这样，就可以实现组件之间的按需通信。

### 访问者模式

```cpp
struct Message {
    virtual ~Message() = default;
};

struct MoveMessage {
    glm::vec3 velocityChange;
};

struct JumpMessage {
    double jumpHeight;
};
```

如何定义对所有不同类型消息的处理方式？

```cpp
struct MessageVisitor;  // 前向声明

struct Message {
    virtual void accept(MessageVisitor *visitor) = 0;
    virtual ~Message() = default;
};

struct MoveMessage {
    glm::vec3 velocityChange;

    void accept(MessageVisitor *visitor) override {
        visitor->visit(this);  // 会调用到 visit(MoveMessage *mm) 这个重载
    }
};

struct JumpMessage {
    double jumpHeight;

    void accept(MessageVisitor *visitor) override {
        visitor->visit(this);  // 会调用到 visit(JumpMessage *mm) 这个重载
    }
};

struct MessageVisitor {
    virtual void visit(MoveMessage *mm) {}  // 默认不做任何处理
    virtual void visit(JumpMessage *jm) {}  // 默认不做任何处理
};

struct Movable : MessageVisitor {
    glm::vec3 position;
    glm::vec3 velocity;

    void handleMessage(Message *msg) {
        msg->accept(this);
    }

    void visit(MoveMessage *mm) override {
        velocity += mm->velocityChange;
    }

    void visit(JumpMessage *jm) override {
        velocity.y += sqrt(2 * 9.8 * jm->jumpHeight);
    }
};
```

这就是访问者模式，同时用到了面向对象的虚函数和重载机制，实现了对所有不同类型消息都能定制一个处理方式，而不用通过低效的 `dynamic_cast` 判断消息类型。

访问者模式是否符合开闭原则呢？

当我们新增一种消息类型时，需要修改的地方有：

1. 新增消息类型
2. 在 `MessageVisitor` 中添加一个 `visit` 的重载

当我们新增一种组件类型时，需要修改的地方有：

1. 新增组件类型

这三项修改都是符合开闭原则的，并不会出现牵一发而动全身的情况。

但每个组件都要处理所有消息，这就是一个不符合开闭原则的设计，因此我们让所有的 visit 虚函数有一个默认实现，那就是什么都不做。这样当新增消息类型时，虽然需要每个组件都重新编译了，但是程序员无需修改任何代码，源码级别上，是满足开闭原则的。

访问者模式通常用于 acceptor 数量有限，但 visitor 的组件类型千变万化的情况。

- 如果消息类型有限，组件类型可能经常增加，那需要把组件类型作为 visitor，消息类型作为 acceptor。
- 如果组件类型有限，消息类型可能经常增加，那需要把消息类型作为 visitor，组件类型作为 acceptor。

- 常作为 acceptor 的有：编译器开发中的 IR 节点（代码中间表示），游戏与 UI 开发中的消息类型。
- 常作为 visitor 的有：编译器开发中的优化 pass（会修改 IR 节点），游戏与 UI 开发中的接受消息组件类型。

但是每个组件都要实现 `accept` 的重载，内容完全一样，出现了代码重复。

Java 的模板是 type-erasure 的，对此束手无策。而 C++ 的模板是 refined-generic，可以利用 CRTP 自动实现这部分：

```cpp
struct Message {
    virtual void accept(MessageVisitor *visitor) = 0;
    virtual ~Message() = default;
};

template <class Derived>
struct MessageImpl : Message {
    void accept(MessageVisitor *visitor) override {
        static_assert(std::is_base_of_v<MessageImpl, Derived>);
        visitor->visit(static_cast<Derived *>(this));
    }
};

struct MoveMessage : MessageImpl<MoveMessage> {
    glm::vec3 velocityChange;
    // 自动实现了 accept 函数
};

struct JumpMessage : MessageImpl<JumpMessage> {
    double jumpHeight;
};
```

> 在小彭老师自主研发的 Zeno 中，ZFX 编译器的 IR 优化系统就运用了 CRTP 加持的访问者模式。

## MVC 模式

设计模式是一个巨大的话题，本期先讲到这里，下集我们继续介绍 UI 开发中大名鼎鼎的 MVC 模式。

MVC 模式是一种架构模式，它将应用程序分为三个核心部分：模型（Model）、视图（View）和控制器（Controller），通过分离应用程序的输入、处理和输出来提高应用程序的可维护性和可扩展性。

- 模型（Model）：负责处理数据和业务逻辑，通常由数据结构和数据库组成。
- 视图（View）：负责展示数据和用户界面，通常由 HTML、CSS 和 JavaScript 组成。
- 控制器（Controller）：负责处理用户交互和调度模型和视图，通常由后端语言（如 PHP、Java 等）实现。

MVC 模式的优点：

- 低耦合：模型、视图和控制器之间的职责清晰，可以更容易地进行单独的修改和维护。
- 可扩展性：由于模型、视图和控制器之间的低耦合性，可以更容易地添加新的功能和组件。
- 可维护性：分离了不同的职责，使得代码更容易理解和维护。
