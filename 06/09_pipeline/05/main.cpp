#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include "ticktock.h"
#include <tbb/parallel_pipeline.h>

struct Data {
    std::vector<float> arr;

    Data() {
        arr.resize(std::rand() % 100 * 500 + 10000);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = std::rand() * (1.f / (float)RAND_MAX);
        }
    }

    void step1() {
        for (int i = 0; i < arr.size(); i++) {
            arr[i] += 3.14f;
        }
    }

    void step2() {
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size() - 1; i++) {
            tmp[i] = arr[i - 1] + arr[i] + arr[i + 1];
        }
        std::swap(tmp, arr);
    }

    void step3() {
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = std::sqrt(std::abs(arr[i]));
        }
    }

    void step4() {
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size() - 1; i++) {
            tmp[i] = arr[i - 1] - 2 * arr[i] + arr[i + 1];
        }
        std::swap(tmp, arr);
    }
};

int main() {
    size_t n = 1<<11;

    std::vector<Data> dats(n);
    std::vector<float> result;

    TICK(process);
    auto it = dats.begin();
    tbb::parallel_pipeline(8
    , tbb::make_filter<void, Data *>(tbb::filter_mode::serial_in_order,
    [&] (tbb::flow_control &fc) -> Data * {
        if (it == dats.end()) {
            fc.stop();
            return nullptr;
        }
        return &*it++;
    })
    , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
    [&] (Data *dat) -> Data * {
        dat->step1();
        return dat;
    })
    , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
    [&] (Data *dat) -> Data * {
        dat->step2();
        return dat;
    })
    , tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
    [&] (Data *dat) -> Data * {
        dat->step3();
        return dat;
    })
    , tbb::make_filter<Data *, float>(tbb::filter_mode::parallel,
    [&] (Data *dat) -> float {
        float sum = std::reduce(dat->arr.begin(), dat->arr.end());
        return sum;
    })
    , tbb::make_filter<float, void>(tbb::filter_mode::serial_out_of_order,
    [&] (float sum) -> void {
        result.push_back(sum);
    })
    );
    TOCK(process);

    return 0;
}
