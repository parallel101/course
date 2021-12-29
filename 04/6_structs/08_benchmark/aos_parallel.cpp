#include "common.h"

namespace aos_parallel {

struct Point {
    float x;
    float y;
    float z;
};

Point ps[N];

void compute() {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        ps[i].x = ps[i].x + ps[i].y + ps[i].z;
    }
}

}
