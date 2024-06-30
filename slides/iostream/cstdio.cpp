#include <cstdio>
#include <thread>

using namespace std;

// Linux:
// stdout: _IOLBF 行缓冲
// stderr: _IONBF 无缓冲
// fopen: _IOFBF 完全缓冲

// MSVC:
// stdout: _IONBF 无缓冲
// stderr: _IONBF 无缓冲
// fopen: _IOFBF 完全缓冲

static char buf[BUFSIZ];

int main() {
    setvbuf(stdout, buf, _IOFBF, sizeof buf);

    for (int i = 0; i < 65536; i += 8) {
        fprintf(stdout, "[%5d]\n", i);
        this_thread::sleep_for(1ms);
    }

    return 0;
}
