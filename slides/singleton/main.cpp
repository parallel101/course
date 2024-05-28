#include "debug.hpp"

using namespace std;

struct Game {
    Game() {
        puts("构造函数");
    }

    ~Game() {
        puts("析构函数");
    }
};

Game &getGame() {
    static Game game;
    return game;
}

int main() {
    puts("main准备调用getGame");
    getGame();
    puts("main调用完了getGame");

    puts("main准备再次调用getGame");
    getGame();
    puts("main再次调用完了getGame");
    return 0;
}
