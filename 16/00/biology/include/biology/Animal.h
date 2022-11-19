#pragma once

#include <ostream>

namespace biology {

struct Animal {
    virtual void speak(std::ostream &os) const = 0;
    virtual ~Animal() = default;
};

struct Cat : Animal {
    virtual void speak(std::ostream &os) const override;
};

struct Dog : Animal {
    virtual void speak(std::ostream &os) const override;
};

}
