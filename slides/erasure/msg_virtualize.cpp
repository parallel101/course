#include <memory>
#include <vector>
#include "msglib.h"

struct MsgBase {
    virtual void speak() = 0;
    virtual void happy() = 0;
    virtual std::shared_ptr<MsgBase> clone() const = 0;
    virtual ~MsgBase() = default;
};

template <class Msg>
struct MsgImpl : MsgBase {
    Msg msg;

    template <class ...Ts>
    MsgImpl(Ts &&...ts) : msg{std::forward<Ts>(ts)...} {
    }

    void speak() override {
        msg.speak();
    }

    std::shared_ptr<MsgBase> clone() const override {
        return std::make_shared<MsgImpl<Msg>>(msg);
    }
};

template <class Msg, class ...Ts>
std::shared_ptr<MsgBase> makeMsg(Ts &&...ts) {
    return std::make_shared<MsgImpl<Msg>>(std::forward<Ts>(ts)...);
}

struct MsgFactoryBase {
    virtual std::shared_ptr<MsgBase> makeMsg() = 0;
};

template <class Msg>
struct MsgFactoryImpl {
};

std::vector<std::shared_ptr<MsgBase>> msgs;

int main() {
    msgs.push_back(makeMsg<MoveMsg>(5, 10));
    msgs.push_back(makeMsg<JumpMsg>(20));
    msgs.push_back(makeMsg<SleepMsg>(8));
    msgs.push_back(makeMsg<ExitMsg>());

    for (auto &msg : msgs) {
        msg->speak();
    }

    return 0;
}
