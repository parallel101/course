#include <deque>
#include <list>
#include <set>
#include <thread>
#include <unordered_set>
#include <vector>
#include <tsl/robin_set.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/node_hash_set.h>

using namespace std::chrono_literals;

int main() {
    { // vector
        std::vector<int> a;
        for (int i = 0; i < 400; i++) {
            a.push_back(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // deque
        std::deque<int> a;
        for (int i = 0; i < 400; i++) {
            a.push_back(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // list
        std::list<int> a;
        for (int i = 0; i < 100; i++) {
            a.push_back(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // set
        std::set<int> a;
        for (int i = 0; i < 100; i++) {
            a.insert(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // unordered_set
        std::unordered_set<int> a;
        for (int i = 0; i < 100; i++) {
            a.insert(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // robin_set
        tsl::robin_set<int> a;
        for (int i = 0; i < 100; i++) {
            a.insert(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // flat_hash_set
        absl::flat_hash_set<int> a;
        for (int i = 0; i < 100; i++) {
            a.insert(i);
            std::this_thread::sleep_for(1us);
        }
    }
    { // node_hash_set
        absl::node_hash_set<int> a;
        for (int i = 0; i < 100; i++) {
            a.insert(i);
            std::this_thread::sleep_for(1us);
        }
    }
    return 0;
}
