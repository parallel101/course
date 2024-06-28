#include <ios>
#include <iostream>
#include <string>
#include <memory>
#include <sstream>
#include <json/json.h>
#include "reflect.hpp"


struct Student {
    std::string name;
    int age;
    int sex;

    void speak(std::string lang, bool newline) const {
        if (lang == "cn") {
            std::cout << "你好！我是" << name << "，今年" << age << "岁，是" << (sex == 1 ? "男" : "女") << "生。";
        } else {
            std::cout << "Hello! I am " << name << ", " << age << " years old, a " << (sex == 1 ? "boy" : "girl") << ".";
        }
        if (newline) {
            std::cout << '\n';
        }
    }

    REFLECT(name, age, sex, speak); // 就地定义
};


struct Student2 {
    std::string name;
    int age;
    int sex;
};
REFLECT_TYPE(Student2, name, age, sex); // 如果不能修改类的定义


template <class T, class N>
struct Baby {
    N name;
    T hungry;
};
REFLECT_TYPE_TEMPLATED(((Baby<T, N>), class T, class N), name, hungry); // 如果不能修改类的定义，且类是模板

template <class T>
std::string jsonSerialize(T &object) {
    Json::Value root;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        root[key] = value;
    });
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ostringstream os;
    writer->write(root, &os);
    return os.str();
}

template <class T>
T jsonDeserialize(std::string const &json) {
    Json::Value root;
    Json::Reader reader;
    reader.parse(json, root);
    T object;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        value = root[key].as<std::decay_t<decltype(value)>>();
    });
    return object;
}


int main() {
    Student stu = {
        .name = "Peng",
        .age = 23,
        .sex = 1,
    };
    Student2 stu2 = {
        .name = "Sir",
        .age = 23,
        .sex = 1,
    };
    Baby<int, std::string> baby = {
        .name = "Tom",
        .hungry = 1,
    };
    Baby<float, const char *> baby2 = {
        .name = "Jerry",
        .hungry = 2.0f,
    };

    std::string bin = jsonSerialize(stu);
    std::cout << bin << '\n';
    auto stuDes = jsonDeserialize<Student>(bin);
    std::cout << stuDes.name << '\n';
    std::cout << stuDes.age << '\n';
    std::cout << stuDes.sex << '\n';

    std::cout << std::boolalpha;
    std::cout << reflect::get_member<std::string>(stu, "name") << '\n';
    std::cout << reflect::get_member<int>(stu, "age") << '\n';
    std::cout << reflect::get_member<int>(stu, "sex") << '\n';
    std::cout << reflect::has_member<Student>("sex") << '\n';
    std::cout << reflect::has_member<Student>("sey") << '\n';
    std::cout << reflect::is_member_type<Student, int>("sex") << '\n';
    std::cout << reflect::is_member_type<Student, std::string>("sex") << '\n';
    std::cout << reflect::is_member_type<Student, const int>("sex") << '\n';
    // reflect::get_function<void(std::string, bool)>(stu, "speak")("cn", true);

    bin = jsonSerialize(stu);
    std::cout << bin << '\n';
    bin = jsonSerialize(baby);
    std::cout << bin << '\n';
    bin = jsonSerialize(baby2);
    std::cout << bin << '\n';

    return 0;
}
