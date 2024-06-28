#include <iostream>
#include <pybind11/pybind11.h>
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

template <class T>
pybind11::class_<T> autodef(pybind11::module &m, const char *clsname) {
    pybind11::class_<Address> c(m, clsname);
    reflect::foreach_member_ptr<Address>([&] (const char *name, auto member) {
        c.def(name, member);
    });
    return c;
}

PYBIND11_MODULE(pyreflect, m) {
    autodef<Address>(m, "Address");
    autodef<Student>(m, "Student");
}
