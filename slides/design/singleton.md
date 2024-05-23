# 游戏开发中可能用到的设计模式

- 单例模式
- 模板模式
- 状态模式
- 组件模式
- 观察者模式
- 发布-订阅模式
- 访问者模式
- 原型模式
- CRTP 模式
- P-IMPL 模式
- 桥接模式

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
    static Game instance;
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
    static Game &instance() {
        static Game game;
        return game;
    }
};

Game::instance().updatePlayers();
```

### 通用的单例模式模板

```cpp
template <class T>
T &singleton() {
    static T inst;
    return inst;
}

singleton<Game>().updatePlayers();
```

## 模板模式

用于游戏中一些相似的处理逻辑，把共同的部分抽象到一个基类，把不同的部分留给派生类实现。

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
    virtual void draw() = 0;
    virtual void move() = 0;

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

## 状态模式

游戏中的角色通常有多种状态，例如，一个怪物可能有“待机”、“巡逻”、“追击”、“攻击”等多种状态，而每种状态下的行为都不一样。

状态模式可以把每种状态抽象为一个类，让角色持有当前状态，而不必每次都通过 if 判断来执行不同的行为。

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

一些糟糕的开发者会使用枚举来表示状态的迁移，然后每次都用 switch 来处理不同的状态。

这在性能上有一定优势，缺点是，每次都需要通过 switch 来判断当前状态，添加一个新状态时需要修改所有函数，不符合开闭原则。

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

## 组件模式

游戏中的物体（游戏对象）通常由多个组件组成，例如，一个角色可能由“角色控制器”、“角色外观”、“角色动画”等组件组成，一个子弹可能由“子弹物理”、“子弹外观”等组件组成。

组件模式是游戏领域最重要的设计模式，它将游戏对象分为多个组件，每个组件只关心自己的逻辑，而不关心其他组件的逻辑。

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
    virtual ~Component() = default;  // 注意
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
    unordered_map<type_index, vector<Component *>> subscribers;

    template <class EventType>
    void subscribe(Component *component) {
        subscribers[type_index(typeid(EventType))].push_back(component);
    }

    template <class EventType>
    void send(EventType *msg) {
        for (auto &&c: subscribers[type_index(typeid(T))]) {
            c->handleMessage(msg);
        }
    }

    void add(Component *component) {
        components.push_back(component);
        component->subscribeMessages();
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
