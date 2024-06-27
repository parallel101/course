#include "func.h"
#include <cstdio>

void func(Color color) {
    switch (color) {
    case RED: printf("RED\n"); [[fallthrough]];
    case YELLOW: printf("YELLOW\n"); break;
    case GREEN: printf("GREEN\n"); break;
    case BLUE: printf("BLUE\n"); break;
    }
    int icolor = static_cast<int>(color);
    printf("%d\n", icolor);
}
