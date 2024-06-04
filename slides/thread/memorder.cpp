#include "mtverify.hpp"

using namespace std;

struct Test {
    atomic_int a_ok;
    int a;
    /* int a_ok; */
    int r;

    void t1() {
        a = 1;
        a_ok.store(1, memory_order_relaxed);
        /* a_ok = 1; */
    }

    void t2() {
        while (!a_ok.load(memory_order_relaxed));
        /* while (!a_ok); */
        r = a;
    }

    auto repr() {
        return r;
    }
};

int main() {
    mtverify({&Test::t1, &Test::t2});
    return 0;
}
