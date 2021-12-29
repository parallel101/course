#include "common.h"

namespace aos_aligned {

struct Point {
    float x;
    float y;
    float z;
    char padding[4];
};

Point ps[N];

void compute() {
    for (int i = 0; i < N; i++) {
        ps[i].x = ps[i].x + ps[i].y + ps[i].z;
    }
}

}
