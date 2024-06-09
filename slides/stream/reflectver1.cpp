#include <criterion/criterion.h>
#include <sstream>
#include <string>

template <class T>
struct reflect_traits;

template <class T>
struct reflect_traits<T const> {
    template <class Visitor>
    static void for_each_member(T const &object, Visitor &&visitor) {
        return reflect_traits<T>::for_each_member(const_cast<T &>(object), [&] (const char *name, auto const &member) {
            return visitor(name, member);
        });
    }
};

#define R_BEGIN(Type, ...) \
template <__VA_ARGS__> \
struct reflect_traits<Type> { \
    template <class Visitor> \
    static void for_each_member(Type &object, Visitor &&visitor) {
#define R(member) \
        visitor(#member, object.member)
#define R_END() \
    } \
};

template <class T>
struct reflect_t {
    T &object;

    reflect_t(T &object) : object(object) {}

    template <class Visitor>
    void for_each_member(Visitor &&visitor) {
        reflect_traits<T>::for_each_member(object, visitor);
    }

    template <class Type>
    Type &get_member(const char *name) {
        Type *member_p = nullptr;
        reflect_traits<T>::for_each_member(object, [&] (const char *got_name, auto &member) {
            if (got_name == name) {
                if constexpr (std::is_same_v<std::decay_t<decltype(member)>, Type>) {
                    member_p = &member;
                } else {
                    throw std::invalid_argument("member type mismatch");
                }
            }
        });
        if (!member_p) throw std::invalid_argument("no such member");
        return *member_p;
    }
};

template <class T>
reflect_t<T> reflect(T &object) {
    return object;
}

// ===== TEST ZONE =====

struct Student {
    std::string name;
    int age;
};

R_BEGIN(Student);
R(name);
R(age);
R_END();

Test(reflect, for_each_member) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    std::ostringstream os;
    reflect(stu).for_each_member([&] (const char *name, auto &member) {
        os << name << ": " << member << '\n';
    });
    cr_expect_eq(os.str(), "name: 彭于斌\nage: 23\n");
    reflect(stu).for_each_member([&] (const char *name, auto &member) {
        if constexpr (std::is_same_v<std::decay_t<decltype(member)>, int>) {
            cr_expect_str_eq(name, "age");
            member = 10;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(member)>, std::string>) {
            cr_expect_str_eq(name, "name");
            member = "mq白律师";
        }
    });
    cr_expect_eq(stu.name, "mq白律师");
    cr_expect_eq(stu.age, 10);
}

Test(reflect, for_each_member_const) {
    const Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    std::ostringstream os;
    reflect(stu).for_each_member([&] (const char *name, auto &member) {
        os << name << ": " << member << '\n';
    });
    cr_expect_eq(os.str(), "name: 彭于斌\nage: 23\n");
}

Test(reflect, get_member) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    cr_expect_eq(reflect(stu).get_member<std::string>("name"), stu.name);
    reflect(stu).get_member<std::string>("name") = "mq白律师";
    cr_expect_eq("mq白律师", stu.name);
}

Test(reflect, get_member_const) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    cr_expect_eq(reflect(stu).get_member<std::string>("name"), stu.name);
}
