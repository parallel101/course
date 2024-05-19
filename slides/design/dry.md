## 设计模式的前世今生

所有眼花缭乱的设计模式，不是为了什么“高雅”、“可读性”、“讨好面试官”

只为一件事：适应变化！

软件和画画等领域有个显著不同：

画画通常是一次性完成的，但软件的开发是持续性的。

需求经常临时变化，在好几年的跨度中不断地修改和更新，以适应变化的需求，停止更新的软件等于死亡。

比如 B 站以前只支持传视频，但是后来产生了直播、专栏、互动视频等多种需求，最近还引入了充电视频、充电动态等私有制特色功能，这都需要程序员加班加点去更新。

调查显示，程序员 90% 的时间都花在改程序上！设计模式的目的，就是让软件更容易适应变化，牺牲一点写程序的时间，让以后改程序能更轻松。

## 什么情况下不需要设计模式？

**设计模式是为了增援未来！如果你都没有未来...**

鼓捣设计模式的代价是第一次写的时候会辛苦一点，但是后续更新起来就方便了。

就好比你来到一间宿舍，如果你只打算住一天，那么你可以只管拉答辩，而不抽马桶！破罐子破摔！
但现实情况往往是，这个宿舍你需要用很久，那就得注意卫生，及时抽掉马桶，不要给明天的自己添乱。

> 不建议现实中这样做，因为会有人被迫接手你的宿舍（项目）替你擦屁股。

如果你的代码是一次性的，比如 ACM 算法竞赛提交完代码就走人，**不需要后续更新**。那就可以不用设计模式，随地答小辩，反正自己不用负责收场。

**设计模式是为了先苦后甜！结果你后面还是苦？**

如果你为了“优雅强迫症”，用了一堆花里胡哨的设计模式，整个代码已经复杂得牵一发而动全身。
第一次写的时候为了设计模式弄得很辛苦，最后发现后续改起来还是辛苦！
那说明你可能**用错了设计模式**，最终并没有起到缓解未来压力的效果，违背了设计模式的初衷！还不如一开始就不要设计模式。

使用了错误的设计模式就好比你拉完答辩，但是你却去擦镜子，擦了半天辛苦死了，但是第二天依然臭气扑鼻，因为你没根据可能的变化，用对设计模式。

## 来看案例吧！

你做了一个名叫 Alice 的项目，为了输出日志，你是这样写的：

```cpp
log("爱丽丝计划正在启动！", title = "Alice");
log("发生了寒武纪大爆发！", title = "Alice");
log("这是一条提示信息！",   title = "Alice");
```

这里面 title 是每条日志的标题，打印在终端上时会变成这样：

```
14:58:01 [Alice] 爱丽丝计划项目正在启动！
14:58:15 [Alice] 发生了 xxx 事件！
14:58:20 [Alice] 这是一条提示信息！
```

这就出现了讨厌的重复。

### 面向过程

为了避免重复，你封装了一个函数：

```cpp
void mylog(string msg) {
    log(msg, title = "Alice");
}
```

```cpp
mylog("爱丽丝计划正在启动！");
mylog("发生了寒武纪大爆发！");
mylog("这是一条提示信息！");
```

再也不用重复指定一模一样的 title 参数啦！
运行效果相同，写起来却更轻松，看起来也更简洁明了。

这就是**面向过程**的设计模式，用函数把相似的操作封装起来。

实际上你用 log 本身已经在用别人的封装了，如果你把 log 内部的层层调用全部展开，变成原始的一行行系统调用...

```cpp
char buf[1024];
size_t top;
if (top < 1024) buf[top++] = "爱"; else write(1, buf, top), top = 0;
if (top < 1024) buf[top++] = "丽"; else write(1, buf, top), top = 0;
if (top < 1024) buf[top++] = "丽"; else write(1, buf, top), top = 0;
if (top < 1024) buf[top++] = "丝"; else write(1, buf, top), top = 0;
if (top < 1024) buf[top++] = "计"; else write(1, buf, top), top = 0;
if (top < 1024) buf[top++] = "划"; else write(1, buf, top), top = 0;
...
```

可以毫不夸张的说，一切程序，都是在消除重复！
C 编译器通过代码转译消除了汇编的重复，C 标准库通过封装函数消除了常用操作的重复，浏览器通过 HTML/CSS/JS 消除了 UI 开发的重复。

### 为什么复制粘贴不是个好习惯？

你会说，我复制粘贴不就好了？先复制一大堆：

然后选中前面的 `"爱丽丝计划正在启动！"` 逐个修改为不同的信息。

```cpp
log("爱丽丝计划正在启动！", title = "Alice");
log("发生了寒武纪大爆发！", title = "Alice");
log("这是一条提示信息！",   title = "Alice");
```

---

然而避免重复并不是为了少打代码！

假如你需要给项目改个名字，不叫 Alice 了，叫 AlanWalker 了，如果不用设计模式，那你需要手忙脚乱地连改三个地方：

```cpp
log("爱丽丝计划正在启动！", title = "AlanWalker");
log("发生了寒武纪大爆发！", title = "AlanWalker");
log("这是一条提示信息！",   title = "AlanWalker");
```

而且万一其中一个拼错了，然后这个拼错的地方还是一个小概率分支，那么很有可能测试时没有发现就上线了，埋下定时炸弹：

```cpp
log("爱丽丝计划正在启动！", title = "AlanWalker");

if (random() < 0.0001) { // 模拟小概率事件
    log("发生了寒武纪大爆发！", title = "AlanWalkor"); // 名字写错了！
    // 由于这一行代码很少执行到，平时测试时你根本没发现
    // 或者干脆间歇性崩溃，你半天找不到原因所在而抓狂
}

log("这是一条提示信息！",   title = "AlanWalker");
```

面向过程设计模式允许我们把相同的东西抽离出来，相似的东西集中到一块，使你只需要修改一个地方即可：

```cpp
void mylog(string msg) {
    log(msg, title = "AlanWalker");  // 只需这里一处修改，处处生效！
}

mylog("爱丽丝计划正在启动！");
mylog("发生了寒武纪大爆发！");
mylog("这是一条提示信息！");
```

而且相关的事物集中起来后，方便你检查，如果写错了更容易发现，不用一个个去大范围查找所有的 AlanWalker 看看是不是都写对了。

### 面向过程的困境

现在 Alice 需要和他的一个老朋友 Bob 合作：

```cpp
void alice() {
    log("A 项目正在启动！",     title = "Alice");
    log("发生了寒武纪大爆发！", title = "Alice");
    log("这是一条提示信息！",   title = "Alice");
    log("发生了私聊事件！",     title = "Alice");
}

void bob() {
    log("B 项目正在启动！",     title = "Bob");
    log("这是一条好友请求！",   title = "Bob");
    log("结交了新朋友！",       title = "Bob");
}
```

如果还是只有一个 mylog 函数的话，要么 Alice 能够方便，但 Bob 就用不了：

```cpp
void mylog(string msg) {
    log(msg, title = "Alice");
}

void alice() {
    mylog("A 项目正在启动！");
    mylog("发生了寒武纪大爆发！");
    mylog("这是一条提示信息！");
    mylog("发生了私聊事件！");
}

void bob() {
    // Bob: 你方便了，我呢？
    log("B 项目正在启动！",     title = "Bob");
    log("这是一条好友请求！",   title = "Bob");
    log("结交了新朋友！",       title = "Bob");
}
```

这时如果 Bob 要改名，还是需要手忙脚乱！

---

如果分成两个函数：

```cpp
void alicelog(string msg) {
    log(msg, title = "Alice");
}

void boblog(string msg) {
    log(msg, title = "Bob");
}

void alice() {
    alicelog("A 项目正在启动！");
    alicelog("发生了寒武纪大爆发！");
    alicelog("这是一条提示信息！");
    alicelog("发生了私聊事件！");
}

void bob() {
    boblog("B 项目正在启动！");
    boblog("这是一条好友请求");
    boblog("结交了新朋友！");
}
```

且不说这样依然是把两人的名字嵌入了函数名 alicelog 里，要改 "Alice" 时依然需要改 alicelog 的名字。
要是来了个新同学，叫 Carbon，又需要专门为他定义个 carbonlog 函数！根本没起到减少重复的初衷啊！

而且万一你在 alice 函数中不小心打错了一个名字：

```cpp
void alice() {
    alicelog("A 项目正在启动！");
    boblog("发生了寒武纪大爆发！");  // 不小心打错了！但我没注意
    alicelog("这是一条提示信息！");
    alicelog("发生了私聊事件！");
}
```

编译器不会有任何报错，本属于 Alice 的日志就这样以 Bob 的名字写了出去。

可见，**重复性不但会让更新变得困难，还增加了犯错的概率空间**（再次回顾上一集）。

### 初级面向对象 - 封装

```cpp
void alice() {
    log("A 项目正在启动！",     title = "Alice");
    log("发生了寒武纪大爆发！", title = "Alice");
    log("这是一条提示信息！",   title = "Alice");
    log("发生了私聊事件！",     title = "Alice");
}

void bob() {
    log("B 项目正在启动！",     title = "Bob");
    log("这是一条好友请求！",   title = "Bob");
    log("结交了新朋友！",       title = "Bob");
}
```

其实我们可以把两人的名字先存为变量：

```cpp
void alice() {
    auto name = "Alice";
    log("A 项目正在启动！",     title = name);
    log("发生了寒武纪大爆发！", title = name);
    log("这是一条提示信息！",   title = name);
    log("发生了私聊事件！",     title = name);
}

void bob() {
    auto name = "Bob";
    log("B 项目正在启动！",     title = name);
    log("这是一条好友请求！",   title = name);
    log("结交了新朋友！",       title = name);
}
```

这样一样实现了集中的效果，如果遇到改项目名的需求，就不用到处跑了。

---

但是假如 log 引入了新的参数 level 呢？
假设 Alice 想要以 INFO 等级输出，Bob 想要以 DEBUG 等级输出。
我们只好在加个 level 变量，应付未来 level 可能需要改变的可能性：

```cpp
void alice() {
    auto name = "Alice";
    auto level = LogLevel::INFO;
    log("A 项目正在启动！",     title = name, level = level);
    log("发生了寒武纪大爆发！", title = name, level = level);
    log("这是一条提示信息！",   title = name, level = level);
    log("发生了私聊事件！",     title = name, level = level);
}

void bob() {
    auto name = "Bob";
    auto level = LogLevel::DEBUG;
    log("B 项目正在启动！",     title = name, level = level);
    log("这是一条好友请求！",   title = name, level = level);
    log("结交了新朋友！",       title = name, level = level);
}
```

现在 log 库又更新了，他们又突然删除了 level 参数，或者说他们给 level 改了个名字，那你又开始手忙脚乱了。

可见，保存 name 变量只解决了 *项目名变更* 的重复问题，而不能解决 *新增参数* 或 *删除参数* 的重复问题。

---

为了让 *新增参数* 或 *删除参数* 时我们也不用手忙脚乱大修改，我们索性把 log 的所有参数封装到一个结构体 LogParams 里：

```cpp
struct LogParams {
    string name;
    LogLevel level;
};

void mylog(string msg, LogParams params) {
    log(msg, title = params.name, level = params.level);
}

void alice() {
    LogParams params = {
        .name = "Alice";
        .level = LogLevel::INFO;
    };
    log("A 项目正在启动！",     params);
    log("发生了寒武纪大爆发！", params);
    log("这是一条提示信息！",   params);
    log("发生了私聊事件！",     params);
}

void bob() {
    LogParams params = {
        .name = "Bob";
        .level = LogLevel::DEBUG;
    };
    log("B 项目正在启动！",     params);
    log("这是一条好友请求！",   params);
    log("结交了新朋友！",       params);
}
```

现在假如 log 库的作者突然更新，加了个 file 参数，用于指定输出到哪个文件。

其他使用了 log 库的用户都骂骂咧咧的，说着 “我测你的码”、“*龙门粗口*” 什么的手忙脚乱改代码去了。

而早有先见之明的你，淡然自若，只是轻轻给结构体添加了一个成员：

```cpp
struct LogParams {
    string name;
    LogLevel level;
    string file = "C:/默认路径.txt";
};

void mylog(string msg, LogParams params) {
    log(msg, title = params.name, level = params.level, file = params.file);
}
```

如果是你编写 Alice 项目，目前尚且只有 3 处调用，你会不会为他专门封装一个 mylog 函数？还是偷懒复制粘贴爽一时？
等发展到有 100 处调用时突然需要改项目名呢？悔恨吗？

如果是你，看到只有一个 title 参数的 log 函数，会不会富有远见地专门为他定义一个尚且只有一个成员的 LogParams 结构体并受到短视人的嘲笑？
等到有 100 处调用时 log 突然需要加个新参数时呢？还笑得出来不？

可能你有 100 处代码提前准备好用了设计模式，即使其中只有 10 个后来确实需要修改，但如果你一开始没用，那这 10 个改起来可能比直接写 1000 个还吃力。

---

后面的发展我们都知道了，业界大量采用上述做法后，发现经常出现一个结构体封装一堆“参数”的情况。

为了方便，人们魔改 C 语言发明 C++ 引入“成员函数”的语法糖来简化书写。
然后结构体也别叫 LogParams 了，直接就叫 Logger 了，“日志记录者”，听起来就好像他是个活人一样！

```cpp
struct Logger {
    string name;
    LogLevel level;

    void mylog(string msg) {
        log(msg, title = this->name, level = this->level);
    }
};

void alice() {
    Logger logger = {
        .name = "Alice";
        .level = LogLevel::INFO;
    };
    logger.mylog("A 项目正在启动！");
    logger.mylog("发生了寒武纪大爆发！");
    logger.mylog("这是一条提示信息！");
    logger.mylog("发生了私聊事件！");
}

void bob() {
    Logger logger = {
        .name = "Bob";
        .level = LogLevel::INFO;
    };
    logger.mylog("B 项目正在启动！");
    logger.mylog("这是一条好友请求！");
    logger.mylog("结交了新朋友！");
}
```

这就是后来风靡一时的面向对象设计模式之雏形，如今绝大多数软件都是基于面向对象开发的。
即使是 C 语言的软件，例如 Linux 内核，也会大量使用这种结构体做参数的方式模拟面向对象，他们甚至还会用函数指针模拟虚函数。
没有面向对象设计模式，任何大型项目都将寸步难行！稍微迈出一步都将轰然倒塌！

而某些象牙塔里眷养着的大学生，毕业设计典中典之图书增删改查系统，经常是图书和学生分别弄四个增删改查函数，完全不复用代码，完全考虑未来增加新实体的可能性。
他们可能用了 Java 语言，思想却是完全面向过程的，和 Linus 用着 C 语言却高度面向对象相比，谁才是真正的面向对象技术的拥护者呢？

他们确实不用考虑未来的维护，毕竟这种应付老师的一次性产品根本不会投入使用！根本不用更新！写完论文直接丢在一旁了！
这就是为什么小彭老师在面试时会发现学历越高越菜的现象，他们大学里只学了**写程序**，而我们的主要工作却是**改程序**！

> 同理 ACM 我建议你少写，里面的纯数学算法大多很少直接用到，用到了也都是调用库，反而写惯了这种随地答辩写法，你回到自己家里也习惯性地答小辩失禁就尴尬了。

### 中级面向对象 - 多态

例如刚刚我们把 log 函数拆开，变成一行行系统调用，这里调用的是 Linux 系统的 write 函数，但假如我们是基于 Windows 系统呢？需要全部改成 WriteConsole 函数！

假如我们突然需求有变，需要写到文件呢？又手忙脚乱复制粘贴改成 WriteFile 函数！
现在要写到一个 TCP 套接字呢？需求没完没了的改，每一次都得推翻重做。
C 标准库好在封装了底层细节，把都统一成了抽象的 fprintf 函数和 FILE 结构体。
把修改全部集中在被调用者（少数）手中，使调用者（多数）不用任何修改就可以自动同时适应 Linux 和 Windows 平台。

但面向过程的 C 语言解决了跨平台（因为跨平台的两份 exe 文件是分别编译的），依然解决不了 TCP 套接字的情况：套接字的读写需要 send/recv 函数，而不是 read/write 函数，需要我们重新写一个底层调用的是 send/recv 的 SOCKET 结构体，尽管他们除了调用函数名外所有逻辑完全一样！
FILE 结构体也解决不了需要把写入到一个字符串而不是文件的情况，只得重新封装了一个 sprintf 函数...
如何才能让同一个类，**不需要重新编译**就能兼具 TCP、字符串多种底层读写实现？

虚函数诞生了。

---

### 功能单一原则

为什么不推荐一个类干很多事：那往往意味着代码出现了重复！

## 初级函数式

面向对象允许一个类有很多个成员函数。

但是多数情况下，其实每个类只需要一个函数就够了。

## 中级函数式 - 回调

## 高级函数式 - 闭包

## 后现代鸭子类型

下一期正式开始逐个详细介绍每一个设计模式，尽情期待～
