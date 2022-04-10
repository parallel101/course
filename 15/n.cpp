#include <string>
#include <iostream>
#include <functional>
#include <memory>
#include <map>

using namespace std;

// main.h
struct Animal {
    Animal() = default;
    Animal(Animal const &) = delete;
    Animal &operator=(Animal const &) = delete;
    virtual ~Animal() = default;

    virtual void eatFood() = 0;
};

map<string, function<unique_ptr<Animal>()>> &getFunctab();

// Cat.cpp
struct Cat : Animal {
    string m_catFood = "someFish";

    virtual void eatFood() override {
        cout << "eating " << m_catFood << endl;
        m_catFood = "fishBones";
    }

    virtual ~Cat() override = default;
};
static int defCat = (getFunctab().emplace("Cat", make_unique<Cat>), 0);

// Dog.cpp
struct Dog : Animal {
    string m_dogFood = "someMeat";

    virtual void eatFood() override {
        cout << "eating " << m_dogFood << endl;
        m_dogFood = "meatBones";
    }

    virtual ~Dog() override = default;
};
static int defDog = (getFunctab().emplace("Dog", make_unique<Dog>), 0);

// main.cpp
map<string, function<unique_ptr<Animal>()>> &getFunctab() {
    static map<string, function<unique_ptr<Animal>()>> inst;
    return inst;
}

int main() {
    unique_ptr<Animal> cat = getFunctab().at("Cat")();
    unique_ptr<Animal> dog = getFunctab().at("Dog")();
    cat->eatFood();
    dog->eatFood();
    return 0;
}
