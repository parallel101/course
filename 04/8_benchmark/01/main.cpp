#include "profile.h"
#include "common.h"

namespace aos {
void compute();
}

namespace aos_aligned {
void compute();
}

namespace aos_parallel {
void compute();
}

namespace soa {
void compute();
}

namespace soa_simd {
void compute();
}

namespace soa_size_t {
void compute();
}

namespace soa_unroll {
void compute();
}

namespace soa_parallel {
void compute();
}

namespace aosoa {
void compute();
}

#if !defined(TIMES)
#define TIMES 10000
#endif

int main() {
    profile(TIMES, "aos", [&] {
        aos::compute();
    });
    profile(TIMES, "aos_aligned", [&] {
        aos_aligned::compute();
    });
    profile(TIMES, "aos_parallel", [&] {
        aos_parallel::compute();
    });
    profile(TIMES, "soa", [&] {
        soa::compute();
    });
    profile(TIMES, "soa_simd", [&] {
        soa_simd::compute();
    });
    profile(TIMES, "soa_size_t", [&] {
        soa_size_t::compute();
    });
    profile(TIMES, "soa_unroll", [&] {
        soa_unroll::compute();
    });
    profile(TIMES, "soa_parallel", [&] {
        soa_parallel::compute();
    });
    profile(TIMES, "aosoa", [&] {
        aosoa::compute();
    });
}
