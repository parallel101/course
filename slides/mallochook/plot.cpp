#include "alloc_action.hpp"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>

struct LifeBlock {
    AllocOp start_op;
    uint32_t tid;
    size_t size;
    void *caller;
    uint64_t start_time;
    uint64_t end_time;
};

struct LifeBlockCompare {
    bool operator()(const LifeBlock &a, const LifeBlock &b) const {
        return a.start_time < b.start_time;
    }
};

void plot_alloc_actions(std::deque<AllocAction> const &actions) {
    std::map<void *, LifeBlock> living;
    std::set<LifeBlock, LifeBlockCompare> dead;
    for (const auto &action : actions) {
        if (kAllocOpIsAllocation[(size_t)action.op]) {
            living.insert({action.ptr, {
                action.op, action.tid, action.size,
                action.caller, action.time, action.time}});
        } else {
            auto it = living.find(action.ptr);
            if (it != living.end()) {
                it->second.end_time = action.time;
            }
            dead.insert(it->second);
            living.erase(it);
        }
    }

    uint64_t start_time = std::numeric_limits<uint64_t>::max();
    uint64_t end_time = std::numeric_limits<uint64_t>::min();
    for (const auto &block : dead) {
        start_time = std::min(start_time, block.start_time);
        end_time = std::max(end_time, block.end_time);
    }

    size_t screen_width = 60;
    auto repeat = [&](uint64_t d, const char *s, const char *end) {
        size_t n = std::max((size_t)d * screen_width, (size_t)0) / (end_time - start_time + 1);
        std::string r;
        for (size_t i = 0; i < n; i++) {
            r += s;
        }
        r += end;
        return r;
    };
    for (const auto &block : dead) {
        std::cout << repeat(block.start_time - start_time, " ", "┌");
        std::cout << repeat(block.end_time - block.start_time, "─", "┐");
        std::cout << block.size << '\n';
    }
}
