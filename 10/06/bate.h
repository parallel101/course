#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <cstdlib>
#include <string>
#include <array>
#include <tuple>
#include <chrono>
#include <map>
#include <unordered_map>
#include <memory>

namespace bate {

static float frand() {
    static std::mt19937 gen;
    static std::uniform_real_distribution<float> unif;
    return unif(gen);
}

static void timing(std::string const &key) {
    static std::map<std::string, std::chrono::steady_clock::time_point> saves;
    auto it = saves.find(key);
    if (it == saves.end()) {
        saves.emplace(key, std::chrono::steady_clock::now());
    } else {
        double dt = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - it->second).count();
        std::cout << key << ": " << dt << "s" << std::endl;
    }
}

}
