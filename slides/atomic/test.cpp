#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef HAS_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <mutex>
#else
#include <algorithm>
#pragma message ("Warning: tbb not found, would be extremely slow")
#endif

using namespace std;

int main() {
    string program = "../build/main";
    size_t repeats = 100;
    system("ulimit -c 0");

    unordered_map<string, size_t> collect;
#ifdef HAS_TBB
    std::mutex collect_mutex;
    tbb::task_arena arena(std::thread::hardware_concurrency() * 6);
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, repeats), [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
#else
    for (size_t i = 0; i < repeats; i++) {
#endif
                FILE *pipe = popen(program.c_str(), "r");
                if (!pipe) {
                    perror(program.c_str());
                    continue;
                }
                char buffer[128];
                string output;
                while (!feof(pipe)) {
                    if (fgets(buffer, sizeof buffer, pipe) != NULL) {
                        output.append(buffer);
                    }
                }
                pclose(pipe);
                while (!output.empty() && output.back() == '\n') {
                    output.pop_back();
                }
                std::lock_guard<std::mutex> lock(collect_mutex);
                ++collect[output];
#ifndef HAS_TBB
    }
#else
            }
        });
    });
#endif

    vector<pair<string, size_t>> results(collect.begin(), collect.end());
    sort(results.begin(), results.end(),
         [](pair<string, size_t> const &a, pair<string, size_t> const &b) {
             return a.second > b.second;
         });

    for (auto &[output, occurances]: results) {
        if (output.find('\n') != output.npos) {
            cout << occurances << "x:\n" << output << "\n\n";
        } else {
            cout << occurances << "x\t" << output << "\n";
        }
    }

    return 0;
}
