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
    virtual shared_ptr<IObject> clone() const = 0;
};

#define IOBJECT_DEFINE_CLONE \
    virtual shared_ptr<IObject> clone() const override { \
        return make_shared<decay_t<decltype(*this)>>(*this); \
    }

struct CatObject : IObject {
    string m_catFood = "someFish";

    IOBJECT_DEFINE_CLONE

    virtual void eatFood() override {
        cout << "eating " << m_catFood << endl;
        m_catFood = "fishBones";
    }

    virtual ~CatObject() override = default;
};

struct DogObject : IObject {
    string m_dogFood = "someMeat";

    IOBJECT_DEFINE_CLONE

    virtual void eatFood() override {
        cout << "eating " << m_dogFood << endl;
        m_dogFood = "meatBones";
    }

    virtual ~DogObject() override = default;
};

void eatTwice(IObject *obj) {
    shared_ptr<IObject> newObj = obj->clone();
    obj->eatFood();
    newObj->eatFood();
}

int main() {
    shared_ptr<CatObject> cat = make_shared<CatObject>();
    shared_ptr<DogObject> dog = make_shared<DogObject>();

    eatTwice(cat.get());
    eatTwice(dog.get());

    return 0;
}
