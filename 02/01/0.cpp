#include <stdlib.h>
#include <stdio.h>

int main() {
    size_t nv = 4;
    int *v = (int *)malloc(nv * sizeof(int));
    v[0] = 4;
    v[1] = 3;
    v[2] = 2;
    v[3] = 1;

    int sum = 0;
    for (size_t i = 0; i < nv; i++) {
        sum += v[i];
    }

    printf("%d\n", sum);

    free(v);
    return 0;
}
