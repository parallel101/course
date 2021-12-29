#include "profile.h"

namespace aos {
void compute();
}

namespace aos_aligned {
void compute();
}

namespace soa {
void compute();
}

namespace soa_size_t {
void compute();
}

namespace soa_unroll {
void compute();
}

namespace aosoa {
void compute();
}

int main() {
    profile(1000, "aos", [&] {
        aos::compute();
    });
    profile(1000, "aos_aligned", [&] {
        aos_aligned::compute();
    });
    profile(1000, "soa", [&] {
        soa::compute();
    });
    profile(1000, "soa_size_t", [&] {
        soa_size_t::compute();
    });
    profile(1000, "soa_unroll", [&] {
        soa_unroll::compute();
    });
    profile(1000, "aosoa", [&] {
        aosoa::compute();
    });
}
