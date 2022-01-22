#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>
#include "ndarray.h"

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t nx = 1<<13;
constexpr size_t ny = 1<<13;
constexpr int nblur = 8;

ndarray<2, float, 16> a(nx, ny);
ndarray<2, float> b(nx, ny);

void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                b(x, y) = a(x, y);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_copy);

void BM_x_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float res = 0;
                for (int t = -nblur; t <= nblur; t++) {
                    res += a(x + t, y);
                }
                b(x, y) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur);

void BM_y_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float res = 0;
                for (int t = -nblur; t <= nblur; t++) {
                    res += a(x, y + t);
                }
                b(x, y) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur);

void BM_y_blur_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int yBase = 0; yBase < ny; yBase += blockSize) {
            for (int xBase = 0; xBase < nx; xBase += blockSize) {
                for (int y = yBase; y < yBase + blockSize; y++) {
                    for (int x = xBase; x < xBase + blockSize; x++) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; t++) {
                            res += a(x, y + t);
                        }
                        b(x, y) = res;
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled);

void BM_y_blur_tiled_only_x(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x++) {
                    float res = 0;
                    for (int t = -nblur; t <= nblur; t++) {
                        res += a(x, y + t);
                    }
                    b(x, y) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x);

void BM_y_blur_tiled_only_x_prefetched(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int xTmp = xBase; xTmp < xBase + blockSize; xTmp += 16) {
                    _mm_prefetch(&a(xTmp, y + nblur), _MM_HINT_T0);
                    for (int x = xTmp; x < xTmp + 16; x++) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; t++) {
                            res += a(x, y + t);
                        }
                        b(x, y) = res;
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched);

void BM_y_blur_tiled_only_x_prefetched_streamed(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int xTmp = xBase; xTmp < xBase + blockSize; xTmp += 16) {
                    _mm_prefetch(&a(xTmp, y + nblur), _MM_HINT_T0);
                    for (int x = xTmp; x < xTmp + 16; x++) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; t++) {
                            res += a(x, y + t);
                        }
                        _mm_stream_si32((int *)&b(x, y), (int &)res);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 16) {
                    _mm_prefetch(&a(x, y + nblur), _MM_HINT_T0);
                    float res[16];
                    for (int offset = 0; offset < 16; offset++) {
                        res[offset] = 0;
                        for (int t = -nblur; t <= nblur; t++) {
                            res[offset] += a(x + offset, y + t);
                        }
                    }
                    for (int offset = 0; offset < 16; offset++) {
                        _mm_stream_si32((int *)&b(x + offset, y), (int &)res);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 16) {
                    _mm_prefetch(&a(x, y + nblur), _MM_HINT_T0);
                    __m128 res[4];
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm_setzero_ps();
                        for (int t = -nblur; t <= nblur; t++) {
                            res[offset] = _mm_add_ps(res[offset],
                                _mm_load_ps(&a(x + offset * 4, y + t)));
                        }
                    }
                    for (int offset = 0; offset < 4; offset++) {
                        _mm_stream_ps(&b(x + offset * 4, y), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 16) {
                    _mm_prefetch(&a(x, y + nblur), _MM_HINT_T0);
                    __m128 res[4];
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm_setzero_ps();
                    }
                    for (int t = -nblur; t <= nblur; t++) {
                        for (int offset = 0; offset < 4; offset++) {
                            res[offset] = _mm_add_ps(res[offset],
                                _mm_load_ps(&a(x + offset * 4, y + t)));
                        }
                    }
                    for (int offset = 0; offset < 4; offset++) {
                        _mm_stream_ps(&b(x + offset * 4, y), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 16) {
                    _mm_prefetch(&a(x, y + nblur), _MM_HINT_T0);
                    __m128 res[4];
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm_setzero_ps();
                    }
                    for (int t = -nblur; t <= nblur; t++) {
#pragma GCC unroll 4
                        for (int offset = 0; offset < 4; offset++) {
                            res[offset] = _mm_add_ps(res[offset],
                                _mm_load_ps(&a(x + offset * 4, y + t)));
                        }
                    }
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; offset++) {
                        _mm_stream_ps(&b(x + offset * 4, y), res[offset]);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x += 32) {
            for (int y = 0; y < ny; y++) {
                _mm_prefetch(&a(x, y + nblur), _MM_HINT_T0);
                _mm_prefetch(&a(x + 16, y + nblur), _MM_HINT_T0);
                __m256 res[4];
#pragma GCC unroll 4
                for (int offset = 0; offset < 4; offset++) {
                    res[offset] = _mm256_setzero_ps();
                }
                for (int t = -nblur; t <= nblur; t++) {
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm256_add_ps(res[offset],
                            _mm256_load_ps(&a(x + offset * 8, y + t)));
                    }
                }
#pragma GCC unroll 4
                for (int offset = 0; offset < 4; offset++) {
                    _mm256_stream_ps(&b(x + offset * 8, y), res[offset]);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x += 32) {
            for (int y = 0; y < ny; y++) {
                _mm_prefetch(&a(x, y + nblur + 40), _MM_HINT_T0);
                _mm_prefetch(&a(x + 16, y + nblur + 40), _MM_HINT_T0);
                __m256 res[4];
#pragma GCC unroll 4
                for (int offset = 0; offset < 4; offset++) {
                    res[offset] = _mm256_setzero_ps();
                }
                for (int t = -nblur; t <= nblur; t++) {
#pragma GCC unroll 4
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm256_add_ps(res[offset],
                            _mm256_load_ps(&a(x + offset * 8, y + t)));
                    }
                }
#pragma GCC unroll 4
                for (int offset = 0; offset < 4; offset++) {
                    _mm256_stream_ps(&b(x + offset * 8, y), res[offset]);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded);

#if 0
void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded_manually(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x += 32) {
            for (int y = 0; y < ny; y++) {
                _mm_prefetch(&a(x, y + nblur + 40), _MM_HINT_T0);
                _mm_prefetch(&a(x + 16, y + nblur + 40), _MM_HINT_T0);
                __m256 res0 = _mm256_load_ps(&a(x + 0, y - nblur));
                __m256 res1 = _mm256_load_ps(&a(x + 8, y - nblur));
                __m256 res2 = _mm256_load_ps(&a(x + 16, y - nblur));
                __m256 res3 = _mm256_load_ps(&a(x + 24, y - nblur));
                for (int t = -nblur; t <= nblur; t++) {
                    res0 = _mm256_add_ps(res0, _mm256_load_ps(&a(x + 0, y + t)));
                    res1 = _mm256_add_ps(res1, _mm256_load_ps(&a(x + 8, y + t)));
                    res2 = _mm256_add_ps(res2, _mm256_load_ps(&a(x + 16, y + t)));
                    res3 = _mm256_add_ps(res3, _mm256_load_ps(&a(x + 24, y + t)));
                }
                _mm256_stream_ps(&b(x + 0, y), res0);
                _mm256_stream_ps(&b(x + 8, y), res1);
                _mm256_stream_ps(&b(x + 16, y), res2);
                _mm256_stream_ps(&b(x + 24, y), res3);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded_manually);

void BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded_manually_saturated(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x += 32) {
            for (int y = 0; y < ny; y++) {
                _mm_prefetch(&a(x, y + nblur + 40), _MM_HINT_T0);
                _mm_prefetch(&a(x + 16, y + nblur + 40), _MM_HINT_T0);
                __m256 res0 = _mm256_load_ps(&a(x + 0, y - nblur));
                __m256 res1 = _mm256_load_ps(&a(x + 8, y - nblur));
                __m256 res2 = _mm256_load_ps(&a(x + 16, y - nblur));
                __m256 res3 = _mm256_load_ps(&a(x + 24, y - nblur));
                res0 = _mm256_add_ps(res0, _mm256_load_ps(&a(x + 0, y - nblur + 1)));
                res1 = _mm256_add_ps(res1, _mm256_load_ps(&a(x + 8, y - nblur + 1)));
                res2 = _mm256_add_ps(res2, _mm256_load_ps(&a(x + 16, y - nblur + 1)));
                res3 = _mm256_add_ps(res3, _mm256_load_ps(&a(x + 24, y - nblur + 1)));
                __m256 res4 = _mm256_load_ps(&a(x + 0, y - nblur + 2));
                __m256 res5 = _mm256_load_ps(&a(x + 8, y - nblur + 2));
                __m256 res6 = _mm256_load_ps(&a(x + 16, y - nblur + 2));
                __m256 res7 = _mm256_load_ps(&a(x + 24, y - nblur + 2));
                for (int t = -nblur + 3; t <= nblur; t += 2) {
                    res0 = _mm256_add_ps(res0, _mm256_load_ps(&a(x + 0, y + t)));
                    res1 = _mm256_add_ps(res1, _mm256_load_ps(&a(x + 8, y + t)));
                    res2 = _mm256_add_ps(res2, _mm256_load_ps(&a(x + 16, y + t)));
                    res3 = _mm256_add_ps(res3, _mm256_load_ps(&a(x + 24, y + t)));
                    res4 = _mm256_add_ps(res4, _mm256_load_ps(&a(x + 0, y + t + 1)));
                    res5 = _mm256_add_ps(res5, _mm256_load_ps(&a(x + 8, y + t + 1)));
                    res6 = _mm256_add_ps(res6, _mm256_load_ps(&a(x + 16, y + t + 1)));
                    res7 = _mm256_add_ps(res7, _mm256_load_ps(&a(x + 24, y + t + 1)));
                }
                res0 = _mm256_add_ps(res0, res4);
                res1 = _mm256_add_ps(res1, res5);
                res2 = _mm256_add_ps(res2, res6);
                res3 = _mm256_add_ps(res3, res7);
                _mm256_stream_ps(&b(x + 0, y), res0);
                _mm256_stream_ps(&b(x + 8, y), res1);
                _mm256_stream_ps(&b(x + 16, y), res2);
                _mm256_stream_ps(&b(x + 24, y), res3);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed_merged_vectorized_interchanged_unrolled_avx_forwarded_manually_saturated);
#endif

BENCHMARK_MAIN();
