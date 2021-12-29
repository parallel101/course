#include "common.h"

namespace aosoa {

#define M (1<<4)

struct Point {
    float x[M];
    float y[M];
    float z[M];
};

Point ps[N / M];

void compute() {
    for (int i = 0; i < N / M; i++) {
        for (int j = 0; j < M; j++) {
            ps[i].x[j] = ps[i].x[j] + ps[i].y[j] + ps[i].z[j];
        }
    }
}

}
