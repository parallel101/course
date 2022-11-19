#include <biology/Carer.h>
#include <biology/Animal.h>
#include <sstream>

namespace biology {

std::string Carer::care(Animal *a) const {
    std::ostringstream ss;
    a->speak(ss);
    return ss.str();
}

}
