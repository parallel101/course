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

template <class Derived, class Base = IObject>
struct IObjectClone : Base {
    virtual shared_ptr<IObject> clone() const override {
        Derived const *that = static_cast<Derived const *>(this);
        return make_shared<Derived>(*that);
    }
};

struct CatObject : IObjectClone<CatObject> {
    string m_catFood = "someFish";

    virtual void eatFood() override {
        cout << "eating " << m_catFood << endl;
        m_catFood = "fishBones";
    }

    virtual ~CatObject() override = default;
};

struct DogObject : IObjectClone<DogObject> {
    string m_dogFood = "someMeat";

    virtual void eatFood() override {
        cout << "eating " << m_dogFood << endl;
        m_dogFood = "meatBones";
    }

    virtual ~DogObject() override = default;
};

template <class AnimalTagT>
struct AnimalObject : IObjectClone<AnimalObject<AnimalTagT>> {
    AnimalTagT m_someTag;
    // ...
};

struct SuperDogObject : IObjectClone<SuperDogObject, DogObject> {
    virtual void eatFood() override {
        DogObject::eatFood();
        cout << "WOOAHHH!!! I'm SUPERDOG~~~!!" << endl;
    }

    virtual ~SuperDogObject() override = default;
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
