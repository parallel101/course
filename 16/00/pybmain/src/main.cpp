#include <iostream>
#include <string>
#include <pybmain/myutils.h>
#include <biology/Animal.h>
#include <biology/Carer.h>

int main() {
    std::cout << "This is biomain!\n";
    biology::Animal *a = new biology::Cat();
    biology::Carer *c = new biology::Carer();
    std::string res = c->care(a);
    res = pybmain::alluppercase(res);
    std::cout << res << '\n';
    delete c;
    delete a;
    return 0;
}
