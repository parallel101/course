#include <benchmark/benchmark.h>
#include <sched.h>
#include <unistd.h>

int main(int argc, char **argv) {
    cpu_set_t cpu;
    CPU_ZERO(&cpu);
    CPU_SET(3, &cpu);
    sched_setaffinity(gettid(), sizeof(cpu), &cpu);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
