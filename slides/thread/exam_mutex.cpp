#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
using namespace std;

string a = "你说的对，因为《高性能并行编程与优化》是小彭老师自主研发的一款 CMake 公开课，后面忘了，总之就是找回失散的“头文件”——同时，逐步发掘“并行”的真相。";
mutex mutex_a;

void t1() {
    std::unique_lock lock(mutex_a);
    a = "你说的对，但我是这个世界的一个和平主义者，我首先收到信息是你们文明的幸运，警告你们：不要回答！不要回答！！不要回答！！！你们的方向上有千万颗恒星，只要不回答，这个世界就无法定位发射源。如果回答，发射源将被定位，你们的文明将遭到入侵，你们的世界将被占领！不要回答！不要回答！！不要回答！！！";
}

void t2() {
    std::unique_lock lock(mutex_a);
    a = "你说的错，因为《原神》是由米哈游自主研发的一款全新开放世界冒险游戏。游戏发生在一个被称作“提瓦特”的幻想世界，在这里，被神选中的人将被授予“神之眼”，导引元素之力。你将扮演一位名为“旅行者”的神秘角色，在自由的旅行中邂逅性格各异、能力独特的同伴们，和他们一起击败强敌，找回失散的亲人——同时，逐步发掘“原神”的真相。";
}

int main() {
    vector<jthread> pool;
    pool.push_back(jthread(t1));
    pool.push_back(jthread(t2));
    pool.clear();
    cout << a << '\n';
    return 0;
}
