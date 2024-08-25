#define TINYBENCH_IMPL
#include "tinybench.hpp"

[[gnu::weak]] int main() {
    std::unique_ptr<tinybench::Reporter> rep(tinybench::makeMultipleReporter({tinybench::makeConsoleReporter()}));
    rep->run_all();
    return 0;
}
