#pragma once

#include <chrono>

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - bench_##x).count() << "ms" << std::endl;
