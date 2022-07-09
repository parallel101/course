#include <memory>
#include <string>
#include <iostream>

using namespace std;

struct IObject {
    IObject() = default;
    IObject(IObject const &) = default;
    IObject &operator=(IObject const &) = default;
    virtual ~IObject() = default;

    virtual void eatFood() = 0;
};

struct CatObject : IObject {
    string m_catFood = "someFish";

    virtual void eatFood() override {
        cout << "cat is eating " << m_catFood << endl;
        m_catFood = "fishBones";
    }

    virtual ~CatObject() override = default;
};

struct DogObject : IObject {
    string m_dogFood = "someMeat";

    virtual void eatFood() override {
        cout << "dog is eating " << m_dogFood << endl;
        m_dogFood = "meatBones";
    }

    virtual ~DogObject() override = default;
};

int main() {
    shared_ptr<CatObject> cat = make_shared<CatObject>();
    shared_ptr<DogObject> dog = make_shared<DogObject>();

    cat->eatFood();
    cat->eatFood();

    dog->eatFood();
    dog->eatFood();

    return 0;
}
