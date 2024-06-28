#include <iostream>
#include <string>
#include <memory>
#include <sstream>
#include <json/json.h>
// #include <pybind11/pybind11.h>
#include "reflect.hpp"


struct Address {
    std::string country;
    std::string province;
    std::string city;

    void show() {
        std::cout << "Country: " << country << '\n';
        std::cout << "Province: " << province << '\n';
        std::cout << "City: " << city << '\n';
    }

    std::string to_str() const {
        return country + " " + province + " " + city;
    }

    static void test() {
        std::cout << "static function test\n";
    }

    REFLECT(country, province, city, show, to_str, test);
};

struct Student {
    std::string name;
    int age;
    Address addr;

    REFLECT(name, age, addr);
};


std::string toString(Json::Value root) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ostringstream os;
    writer->write(root, &os);
    return os.str();
}

Json::Value fromString(std::string const &json) {
    Json::Value root;
    Json::Reader reader;
    reader.parse(json, root);
    return root;
}

template <class T> requires (!reflect_trait<T>::has_member())
Json::Value serialize(T const &object) {
    return object;
}

template <class T> requires (reflect_trait<T>::has_member())
Json::Value serialize(T const &object) {
    Json::Value root;
    foreach_member(object, [&](const char *key, auto &value) {
        root[key] = serialize(value);
    });
    return root;
}

template <class T> requires (!reflect_trait<T>::has_member())
T deserialize(Json::Value const &root) {
    return root.as<T>();
}

template <class T> requires (reflect_trait<T>::has_member())
T deserialize(Json::Value const &root) {
    T object;
    foreach_member(object, [&](const char *key, auto &value) {
        value = deserialize<std::decay_t<decltype(value)>>(root[key]);
    });
    return object;
}

// PYBIND11_MODULE(pyreflect, m) {
//     pybind11::class_<Address> b(m, "Address");
//     reflect_trait<Address>::foreach_member_ptr([&] (const char *name, auto member) {
//         b.def(name, member);
//     });
// }

int main() {
    Student stu = {
        .name = "Peng",
        .age = 23,
        .addr = {
            .country = "China",
            .province = "Shanghai",
            .city = "Shanghai",
        }
    };
    std::string bin = toString(serialize(stu));
    std::cout << bin << '\n';
    auto stuDes = deserialize<Student>(fromString(bin));
    std::cout << stuDes.name << '\n';
    std::cout << stuDes.age << '\n';
    std::cout << stuDes.addr.country << '\n';
    std::cout << stuDes.addr.province << '\n';
    std::cout << stuDes.addr.city << '\n';
    return 0;
}
