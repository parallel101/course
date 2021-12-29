#include "common.h"

namespace soa_unroll {

struct Point {
    float x[N];
    float y[N];
    float z[N];
};

Point ps;

void compute() {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 32
#elif defined(_MSC_VER)
#pragma unroll 32
#endif
    for (int i = 0; i < N; i++) {
        ps.x[i] = ps.x[i] + ps.y[i] + ps.z[i];
    }
}

}
