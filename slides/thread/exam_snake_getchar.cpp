#include "mtqueue.hpp"
#include <iostream>
#include <thread>
#include <termios.h>
#include <vector>
using namespace std;

mt_queue<char> input_queue;

void disable_stdin_buffer() {
    if (isatty(STDIN_FILENO)) {
        struct termios tc;
        tcgetattr(STDIN_FILENO, &tc);
        tc.c_lflag &= ~(tcflag_t)ICANON;
        tc.c_lflag &= ~(tcflag_t)ECHO;
        tcsetattr(STDIN_FILENO, TCSANOW, &tc);
    }
}

void input_thread() {
    disable_stdin_buffer();
    while (true) {
        char c = getchar();
        input_queue.push(c);
    }
}

void game_thread() {
    while (true) {
        if (auto opt_c = input_queue.try_pop_for(500ms)) {
            char c = *opt_c;
            cout << "读到了 " << c << "\n";
        } else {
            cout << "什么都没读到\n";
        }
    }
}

int main() {
    vector<jthread> pool;
    pool.push_back(jthread(game_thread));
    pool.push_back(jthread(input_thread));
    pool.clear();
    return 0;
}
