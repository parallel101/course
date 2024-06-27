#include <iostream>
#include <sstream>
#include <tuple>

struct NodeCommonBase {};

template <class T>
struct ExpressionNode;

template <class Derived>
struct NodeBase : NodeCommonBase {
private:
    Derived const &self() const {
        static_assert(std::is_base_of_v<NodeBase, Derived>, "Derived must derive from NodeBase");
        return static_cast<Derived const &>(*this);
    }

    Derived &self() {
        static_assert(std::is_base_of_v<NodeBase, Derived>, "Derived must derive from NodeBase");
        return static_cast<Derived &>(*this);
    }

public:
#define DEF_NODE_OP_BRACED(opl, opr, opls, oprs) \
    template <class That> \
    auto operator opl opr(That that) const { \
        if constexpr (std::is_base_of_v<NodeCommonBase, That>) { \
            return makeOperationNode([] (auto x, auto y) { return x opl y opr; }, \
                                     std::make_tuple(self(), that), \
                                     self().name() + opls + that.name() + oprs); \
        } else { \
            return self() opl ExpressionNode<That>(that) opr; \
        } \
    }

    DEF_NODE_OP_BRACED([, ], "[", "]")
#define OP_LBRACE (
#define OP_RBRACE )
    DEF_NODE_OP_BRACED(OP_LBRACE, OP_RBRACE, "(", ")")

#undef OP_LBRACE
#undef OP_RBRACE
#undef DEF_NODE_OP_BRACED

#define DEF_NODE_OP2(op) \
    template <class That> \
    auto operator op(That that) const { \
        if constexpr (std::is_base_of_v<NodeCommonBase, That>) { \
            return makeOperationNode([] (auto x, auto y) { return x op y; }, \
                                     std::make_tuple(self(), that), \
                                     self().name() + " " #op " " + that.name()); \
        } else { \
            return self() op ExpressionNode<That>(that); \
        } \
    }

    DEF_NODE_OP2(==)
    DEF_NODE_OP2(!=)
    DEF_NODE_OP2(>)
    DEF_NODE_OP2(<)
    DEF_NODE_OP2(>=)
    DEF_NODE_OP2(<=)
    DEF_NODE_OP2(+)
    DEF_NODE_OP2(-)
    DEF_NODE_OP2(*)
    DEF_NODE_OP2(/)
    DEF_NODE_OP2(^)
    DEF_NODE_OP2(|)
    DEF_NODE_OP2(&)
    DEF_NODE_OP2(%)
    DEF_NODE_OP2(||)
    DEF_NODE_OP2(&&)
    DEF_NODE_OP2(<<)
    DEF_NODE_OP2(>>)

#undef DEF_NODE_OP2

#define DEF_NODE_OP1(op) \
    auto operator op() const { \
        return makeOperationNode([] (auto x) { return op x; }, \
                                 std::make_tuple(self()), \
                                 #op + self().name()); \
    }

    DEF_NODE_OP1(!)
    DEF_NODE_OP1(+)
    DEF_NODE_OP1(-)
    DEF_NODE_OP1(~)

#undef DEF_NODE_OP1

    void base_show() {
        std::ostringstream oss;
        oss << std::boolalpha;
        oss << self().result();
        auto val = oss.str();
        if (val != self().name()) {
            std::cerr << "    \033[33;1m" + self().name()
                + "\033[0m \033[37m=\033[0m \033[35;1m"
                + val + "\033[0m\n";
        }
    }
};

template <class Comp, class ...Ts>
struct OperationNode : NodeBase<OperationNode<Comp, Ts...>> {
    Comp m_comp;
    std::tuple<Ts...> m_args;
    std::string m_name;

    OperationNode(Comp comp, std::tuple<Ts...> args, std::string name)
    : m_comp(comp), m_args(args), m_name(name) {}

    auto result() {
        return std::apply([&] (Ts ...args) {
            return m_comp(args.result()...);
        }, m_args);
    }

    std::string name() const {
        return m_name;
    }

    void show() {
        this->base_show();
        std::apply([] (Ts ...args) {
            int _[]{((void)args.show(), 0)...};
            (void)_;
        }, m_args);
    }
};

template <class Comp, class ...Ts>
auto makeOperationNode(Comp comp, std::tuple<Ts...> args, std::string name) {
    return OperationNode<Comp, Ts...>(comp, args, name);
}

template <class T>
struct ExpressionNode : NodeBase<ExpressionNode<T>> {
    T m_value;
    std::string m_name;

    T result() {
        return m_value;
    }

    std::string name() const {
        return m_name;
    }

    void show() {
        this->base_show();
    }

    ExpressionNode(T value, std::string name) : m_value(value), m_name(name) {
    }

    ExpressionNode(T value) : m_value(value) {
        std::ostringstream oss;
        oss << std::boolalpha;
        oss << result();
        m_name = oss.str();
    }
};

template <class T>
auto makeExpressionNode(T value, std::string name) {
    return ExpressionNode<T>(value, name);
}

struct {
    template <class Node>
    auto operator,(Node node) {
        if constexpr (std::is_base_of_v<NodeCommonBase, Node>) {
            bool success = static_cast<bool>(node.result());
            if (!success) {
                std::cerr << "\033[31;1mexpect failed\033[0m\033[37m:\033[0m \033[33;1m" + node.name() + "\033[0m\n";
                node.show();
                std::abort();
            }
            return *this;
        } else {
            return *this, ExpressionNode<Node>(node);
        }
    }
} expect;

#define Ex(value) makeExpressionNode((value), #value)
#define Ef(func) makeExpressionNode([&] (auto &&..._args) -> decltype(auto) { \
        return func(std::forward<decltype(_args)>(_args)...); \
    }, #func)

int square(int i) {
    return i + i;
}

int main() {
    expect, Ex(square)(2) == 4;
}
