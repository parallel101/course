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

    REFLECT(name, age, sex); // 最简单，最推荐的用法
};


struct Student2 {
    std::string name;
    int age;
    int sex;
};
REFLECT_TYPE(Student2, name, age, sex); // 如果不能加入新成员


template <class T, class N>
struct Baby {
    N name;
    T hungry;
};
REFLECT_TYPE_TEMPLATED(((Baby<T, N>), class T, class N), name, hungry); // 如果不能加入新成员（模板类情况）


template <class T>
std::string serialize(T &object) {
    Json::Value root;
    reflect_trait<T>::for_each_members(object, [&](const char *key, auto &value) {
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
T deserialize(std::string const &json) {
    Json::Value root;
    Json::Reader reader;
    reader.parse(json, root);
    T object;
    reflect_trait<T>::for_each_members(object, [&](const char *key, auto &value) {
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
    std::string bin = serialize(stu);
    std::cout << bin << '\n';
    auto stuDes = deserialize<Student>(bin);
    std::cout << stuDes.name << '\n';
    std::cout << stuDes.age << '\n';
    std::cout << stuDes.sex << '\n';

    bin = serialize(stu);
    std::cout << bin << '\n';
    bin = serialize(baby);
    std::cout << bin << '\n';
    bin = serialize(baby2);
    std::cout << bin << '\n';
    return 0;
}
