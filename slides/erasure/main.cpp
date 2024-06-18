#include <map>
#include <memory>
#include "msglib.h"

struct MsgBase {
    virtual void speak() = 0;
    virtual void load() = 0;
    virtual ~MsgBase() = default;

    using Ptr = std::shared_ptr<MsgBase>;
};

namespace msg_extra_funcs { // 无法为 Msg 们增加成员函数，只能以重载的形式，外挂追加
    void load(MoveMsg &msg) {
        std::cin >> msg.x >> msg.y;
    }

    void load(JumpMsg &msg) {
        std::cin >> msg.height;
    }

    void load(SleepMsg &msg) {
        std::cin >> msg.time;
    }

    void load(ExitMsg &) {
    }
}

template <class Msg>
struct MsgImpl : MsgBase {
    Msg msg;

    void speak() override {
        msg.speak();
    }

    void load() override {
        msg_extra_funcs::load(msg);
    }
};

struct MsgFactoryBase {
    virtual MsgBase::Ptr create() = 0;
    virtual ~MsgFactoryBase() = default;

    using Ptr = std::shared_ptr<MsgFactoryBase>;
};

template <class Msg>
struct MsgFactoryImpl : MsgFactoryBase {
    MsgBase::Ptr create() override {
        return std::make_shared<MsgImpl<Msg>>();
    }
};

template <class Msg>
MsgFactoryBase::Ptr makeFactory() {
    return std::make_shared<MsgFactoryImpl<Msg>>();
}

struct RobotClass {
    inline static const std::map<std::string, MsgFactoryBase::Ptr> factories = {
        {"Move", makeFactory<MoveMsg>()},
        {"Jump", makeFactory<JumpMsg>()},
        {"Sleep", makeFactory<SleepMsg>()},
        {"Exit", makeFactory<ExitMsg>()},
    };

    void recv_data() {
        std::string type;
        std::cin >> type;

        try {
            msg = factories.at(type)->create();
        } catch (std::out_of_range &) {
            std::cout << "no such msg type!\n";
            return;
        }

        msg->load();
    }

    void update() {
        if (msg)
            msg->speak();
    }

    MsgBase::Ptr msg;
};

int main() {
    RobotClass robot;
    robot.recv_data();
    robot.update();
    return 0;
}
