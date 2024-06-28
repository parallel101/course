#include <iostream>
#include "reflect.hpp"
#include "reflect_json.hpp"

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

    std::string binary = reflect_json::serialize(stu);
    std::cout << binary << '\n';
    auto stuDes = reflect_json::deserialize<Student>(binary);

    std::cout << stuDes.name << '\n';
    std::cout << stuDes.age << '\n';
    std::cout << stuDes.addr.country << '\n';
    std::cout << stuDes.addr.province << '\n';
    std::cout << stuDes.addr.city << '\n';

    return 0;
}
