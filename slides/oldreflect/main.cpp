#include <any>
#include <criterion/criterion.h>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>

namespace reflect {

struct member_info {
    const char *name;
    std::size_t offset;
    std::size_t size;
    std::size_t align;
    std::type_info const &type;
};

template <class Class, class Member>
struct typed_member_info : member_info {
    Member Class::*access;

    using member_type = Member;

    template <class T>
    static constexpr std::is_same<T, member_type> is() {
        return {};
    }
};

struct class_meta_info {
    const char *name;
    std::size_t size;
    std::size_t align;
    std::type_info const &type;
};

template <class Class, class ...Members>
struct reflect_info {
    using class_type = Class;

    class_meta_info meta;
    std::tuple<Members...> members;

    template <class F>
    constexpr void for_each_member(F &&f) const {
        std::apply([&] (auto &&...args) {
            (f(args), ...);
        }, members);
    }
};

template <class Class, class ...Members>
constexpr reflect_info<Class, Members...> make_reflect_info(class_meta_info meta, Members ...members) {
    return {meta, std::make_tuple(members...)};
}

template <class T>
struct reflect_traits {
    [[deprecated("reflect trait not declared on this type")]] static constexpr struct {
    } info{};
};

#define REFLECT_CLASS_BEGIN(Type, ...) \
template <__VA_ARGS__> \
struct reflect::reflect_traits<Type> { \
    using class_type = Type; \
    static constexpr auto info = make_reflect_info<class_type>( \
        class_meta_info{ \
                #Type, \
                sizeof(class_type), \
                alignof(class_type), \
                typeid(class_type), \
            }
#define REFLECT_CLASS(member) \
        , typed_member_info<class_type, decltype(class_type::member)>{ \
            { \
                #member, \
                offsetof(class_type, member), \
                sizeof(decltype(class_type::member)), \
                alignof(decltype(class_type::member)), \
                typeid(decltype(class_type::member)), \
            }, \
            &class_type::member, \
        }
#define REFLECT_CLASS_END() \
    ); \
};

template <class T, class F>
constexpr void for_each_member(T &object, F &&f) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    trait_type::info.for_each_member([&] (auto const &member) {
        f(member.name, object.*(member.access));
    });
}

template <class T, class F>
constexpr void for_each_member(F &&f) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    trait_type::info.for_each_member(f);
}

template <class Member, class T>
constexpr std::conditional_t<std::is_const_v<T>, Member const, Member> &get_member(T &object, std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    std::conditional_t<std::is_const_v<T>, Member const, Member> *member_p = nullptr;
    trait_type::info.for_each_member([&] (auto const &member) {
        if (member_p) return;
        if (member.name == name) {
            if constexpr (member.template is<Member>()) {
                member_p = std::addressof(object.*(member.access));
            } else {
                throw std::logic_error("member `" + name + "` has type `" + member.type.name() + "`, given type is `" + typeid(Member).name() + "`");
            }
        }
    });
    if (!member_p)
        throw std::logic_error("no such member named `" + name + "`");
    return *member_p;
}

template <class Any = std::any, class T>
constexpr Any get_member_any(T &object, std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    Any member_any;
    trait_type::info.for_each_member([&] (auto const &member) {
        if (member_any.has_value()) return;
        if (member.name == name) {
            member_any = object.*(member.access);
        }
    });
    return member_any;
}

template <class T, class Member = void>
constexpr bool has_member(std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    enum Status {
        NotFound = 0,
        Found,
        TypeMismatch,
    } status = NotFound;
    trait_type::info.for_each_member([&] (auto member) {
        if (status != NotFound) return;
        if (member.name == name) {
            if constexpr (member.template is<Member>()) {
                status = Found;
            } else {
                status = TypeMismatch;
            }
        }
    });
    if constexpr (std::is_void_v<Member>) {
        return status != NotFound;
    } else {
        return status == Found;
    }
}

template <class T>
constexpr std::type_info const &get_member_type(std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    std::type_info const *member_type = nullptr;
    trait_type::info.for_each_member([&] (auto const &member) {
        if (member_type) return;
        if (member.name == name) {
            member_type = member.type;
        }
    });
    if (!member_type)
        throw std::logic_error("no such member named `" + name + "`");
    return *member_type;
}

template <class T, class Member>
constexpr bool member_type_is(std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    enum Status {
        NotFound = 0,
        Found,
        TypeMismatch,
    } status = NotFound;
    trait_type::info.for_each_member([&] (auto member) {
        if (status != NotFound) return;
        if (member.name == name) {
            if constexpr (member.template is<Member>()) {
                status = Found;
            } else {
                status = TypeMismatch;
            }
        }
    });
    if (status == NotFound)
        throw std::logic_error("no such member named `" + name + "`");
    return status == Found;
}

template <class T>
constexpr member_info get_member_info(std::string const &name) {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    member_info const *member_info = nullptr;
    trait_type::info.for_each_member([&] (auto const &member) {
        if (member_info) return;
        if (member.name == name) {
            member_info = &member;
        }
    });
    if (!member_info)
        throw std::logic_error("no such member named `" + name + "`");
    return *member_info;
}

template <class T>
constexpr class_meta_info get_class_info() {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    return trait_type::info.meta;
}

template <class T>
constexpr std::vector<std::string> get_member_names() {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    std::vector<std::string> member_names;
    trait_type::info.for_each_member([&] (auto const &member) {
        member_names.push_back(member.name);
    });
    return member_names;
}

template <class T>
constexpr std::vector<member_info> get_member_infos() {
    using trait_type = reflect_traits<std::remove_const_t<T>>;
    std::vector<member_info> members;
    trait_type::info.for_each_member([&] (auto const &member) {
        members.push_back(member);
    });
    return members;
}

template <class T>
std::ostream &print(std::ostream &os, T const &object) {
    bool once = false;
    for_each_member(object, [&] (const char *name, auto &member) {
        if (once) {
            os << ", ";
        } else {
            once = true;
        }
        os << name;
        os << ": ";
        print(member);
    });
    os << "}";
    return os;
}

}

// ===== TEST ZONE =====

using namespace reflect;

struct Student {
    std::string name;
    int age;
};

REFLECT_CLASS_BEGIN(Student)
REFLECT_CLASS(name)
REFLECT_CLASS(age)
REFLECT_CLASS_END();

Test(reflect, for_each_member) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    std::ostringstream os;
    for_each_member(stu, [&] (const char *name, auto &member) {
        os << name << ": " << member << '\n';
    });
    cr_expect_eq(os.str(), "name: 彭于斌\nage: 23\n");
    for_each_member(stu, [&] (const char *name, auto &member) {
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
    for_each_member(stu, [&] (const char *name, auto &member) {
        os << name << ": " << member << '\n';
    });
    cr_expect_eq(os.str(), "name: 彭于斌\nage: 23\n");
}

Test(reflect, get_member) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    cr_expect_eq(get_member<std::string>(stu, "name"), stu.name);
    get_member<std::string>(stu, "name") = "mq白律师";
    cr_expect_eq("mq白律师", stu.name);
}

Test(reflect, get_member_const) {
    Student stu{
        .name = "彭于斌",
        .age = 23,
    };
    cr_expect_eq(get_member<std::string>(stu, "name"), stu.name);
}

Test(reflect, static_for_each_member) {
    std::ostringstream os;
    for_each_member<Student>([&] (auto member) {
        os << member.name << ": " << member.offset << '\n';
    });
    cr_expect_eq(os.str(), "name: 0\nage: " + std::to_string(sizeof(std::string)) + "\n");
}

Test(reflect, get_class_members) {
    cr_expect((get_member_names<Student>() == std::vector<std::string>{"name", "age"}));
    cr_expect(get_member_infos<Student>().at(0).name == std::string("name"));
}

Test(reflect, has_member) {
    cr_expect(has_member<Student>("age"));
    cr_expect((has_member<Student, int>("age")));
    cr_expect_not(has_member<Student>("cage"));
    cr_expect_not((has_member<Student, short>("age")));
    cr_expect((has_member<Student, std::string>("name")));
    cr_expect((member_type_is<Student, std::string>("name")));
    cr_expect_not((has_member<Student, const char *>("name")));
    cr_expect_not((member_type_is<Student, const char *>("name")));
    cr_expect(has_member<Student>("name"));
}

consteval auto test() {
    std::unique_ptr<int> p;
    if (&*p == nullptr) {
        return 1;
    } else {
        return 2;
    }
}

inline constexpr auto dummy = test();
