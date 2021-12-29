#include "common.h"

namespace aos {

struct Point {
    float x;
    float y;
    float z;
};

Point ps[N];

void compute() {
    for (int i = 0; i < N; i++) {
        ps[i].x = ps[i].x + ps[i].y + ps[i].z;
    }
}

}
