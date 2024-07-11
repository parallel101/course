#include <print>

struct EatParams {
    int amount;
    int speed;
};

struct DrinkParams {
    int volume;
    int temperature;
};

struct FoodVisitor {
    virtual void visit(struct Eatable *eat) {}
    virtual void visit(struct Drinkable *drink) {}
    virtual ~FoodVisitor() = default;
};

struct Food {
    virtual void accept(FoodVisitor *visitor) = 0;
    virtual ~Food() = default;
};

#define DEF_FOOD_ACCEPT void accept(FoodVisitor *visitor) override { visitor->visit(this); }


struct Drinkable : virtual Food {
    virtual void drink(DrinkParams drinkParams) = 0;

    DEF_FOOD_ACCEPT
};

struct Eatable : virtual Food {
    virtual void eat(EatParams eatParams) = 0;

    DEF_FOOD_ACCEPT
};

struct Cake : Eatable {
    void eat(EatParams eatParams) override {
        std::println("Eating cake...");
        std::println("Amount: {}", eatParams.amount);
        std::println("Speed: {}", eatParams.speed);
    }
};

struct Milk : Drinkable {
    void drink(DrinkParams drinkParams) override {
        std::println("Drinking milk...");
        std::println("Volume: {}", drinkParams.volume);
        std::println("Temperature: {}", drinkParams.temperature);
    }
};

struct Pudding : Eatable, Drinkable {
    void eat(EatParams eatParams) override {
        std::println("Eating pudding...");
        std::println("Amount: {}", eatParams.amount);
        std::println("Speed: {}", eatParams.speed);
    }

    void drink(DrinkParams drinkParams) override {
        std::println("Drinking pudding...");
        std::println("Volume: {}", drinkParams.volume);
        std::println("Temperature: {}", drinkParams.temperature);
    }

    void accept(FoodVisitor *visitor) override {
        Eatable::accept(visitor);
        Drinkable::accept(visitor);
    }
};

struct PengUser : FoodVisitor {
    void visit(Eatable *eat) override {
        eat->eat({5, 10});
    }

    void visit(Drinkable *drink) override {
        drink->drink({10, 20});
    }
};

void pengEat(Food *food) {
    PengUser user;
    food->accept(&user);
    food->accept(&user);
    food->accept(&user);
}

int main() {
    Cake cake;
    Milk milk;
    Pudding pudding;
    pengEat(&cake);
    pengEat(&milk);
    pengEat(&pudding);
    return 0;
}
