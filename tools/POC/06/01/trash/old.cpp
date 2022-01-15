#include <iostream>
#include <variant>
#include <vector>


template <class ...Fs>
struct match : protected Fs... {
    explicit match(Fs const &...fs)
        : Fs(fs)... {}

    using Fs::operator()...;
};

template <class ...Fs>
match(Fs const &...fs) -> match<Fs...>;



/*std::variant<std::false_type, std::true_type>
bool_variant(bool x) {
    if (x)
        return std::true_type{};
    else
        return std::false_type{};
}


void saxpy
    ( std::vector<float> a
    , float b
    , float c
    )
{
    auto has_b = bool_variant(b != 1);
    auto has_c = bool_variant(c != 0);
    std::visit([&] (auto has_b, auto has_c) {
        for (auto &ai: a) {
            if constexpr (has_b)
                ai *= b;
            if constexpr (has_c)
                ai += c;
        }
    }, has_b, has_c);
}*/


struct Course {
    Course() {
        printf("Course构造\n");
    }

    void func() {
        printf("Course的func方法\n");
    }

    void bar() {
        printf("Course的bar方法\n");
    }

    ~Course() {
        printf("Course解构\n");
    }
};

Course &getCourse() {
    static Course course;
    return course;
}

int main() {
    printf("main函数内\n");
    getCourse().func();
    getCourse().bar();
    printf("main函数内\n");
    return 0;
}
