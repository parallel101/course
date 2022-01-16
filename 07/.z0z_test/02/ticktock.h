#pragma once

//#include <chrono>
//#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
//#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count() << "s" << std::endl;

#include <tbb/tick_count.h>

#define TICK(x) auto bench_##x = tbb::tick_count::now();
#define TOCK(x) std::cout << #x ": " << (tbb::tick_count::now() - bench_##x).seconds() << "s" << std::endl;
//#define MTICK(x, times) auto bench_##x = tbb::tick_count::now(); int times_##x = times; for (int count_##x = 0; count_##x < times_##x; count_##x++) {
//#define MTOCK(x) } std::cout << #x ": " << (tbb::tick_count::now() - bench_##x).seconds() / times_##x << "s" << std::endl;
