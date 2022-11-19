#pragma once

#include <string>
#include <cctype>

namespace pybmain {

static std::string alluppercase(std::string s) {
    std::string ret;
    for (char c: s) {
        ret.push_back(std::toupper(c));
    }
    return ret;
}

}
