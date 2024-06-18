#pragma once

#include <iostream>

struct MoveMsg {
    int x;
    int y;

    void speak() {
        std::cout << "Move " << x << ", " << y << '\n';
    }
};

struct JumpMsg {
    int height;

    void speak() {
        std::cout << "Jump " << height << '\n';
    }
};

struct SleepMsg {
    int time;

    void speak() {
        std::cout << "Sleep " << time << '\n';
    }
};

struct ExitMsg {
    void speak() {
        std::cout << "Exit" << '\n';
    }
};
