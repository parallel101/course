#pragma once

enum Color {
    RED,
    GREEN,
    BLUE,
};

void func(Color color);

inline int toint(Color color) {
    return static_cast<int>(color);
}
