#include "profile.h"

namespace aos {
void compute();
}

namespace soa {
void compute();
}

int main() {
    profile(1000, "aos", [&] {
        aos::compute();
    });
    profile(1000, "soa", [&] {
        soa::compute();
    });
}
