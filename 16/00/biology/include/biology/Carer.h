#pragma once

#include <string>

namespace biology {

struct Animal;

struct Carer {
    std::string care(Animal *a) const;
};

}
