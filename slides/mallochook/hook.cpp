#include <chrono>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <new>
#if __unix__
# include <unistd.h>
#elif _WIN32
# include <windows.h>
#endif
#include "alloc_action.hpp"

void plot_alloc_actions(std::deque<AllocAction> const &actions);

namespace {

struct GlobalData {
    bool enable = false;
    std::deque<AllocAction> actions;
    std::mutex lock;

    GlobalData() {
        enable = true;
    }

    ~GlobalData() {
        enable = false;
        plot_alloc_actions(actions);
    }

    void on(AllocOp op, void *ptr, size_t size, size_t align, void *caller) {
        if (ptr) {
#if __unix__
            uint32_t tid = gettid();
#elif _WIN32
            uint32_t tid = GetCurrentThreadId();
#else
            uint32_t tid = 0;
#endif
            auto now = std::chrono::high_resolution_clock::now();
            uint64_t time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch())
                    .count();
            std::lock_guard<std::mutex> guard(lock);
            actions.push_back(
                AllocAction{op, tid, ptr, size, align, caller, time});
        }
    }
} global;

struct EnableGuard {
    bool was_enable;

    EnableGuard() {
        was_enable = global.enable;
        global.enable = false;
    }

    explicit operator bool() {
        return was_enable;
    }

    ~EnableGuard() {
        global.enable = was_enable;
    }
};

} // namespace

void operator delete(void *ptr) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete(void *ptr, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void *operator new(size_t size) noexcept(false) {
    EnableGuard ena;
    void *ptr = malloc(size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, kNone, __builtin_return_address(0));
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void *operator new[](size_t size) noexcept(false) {
    EnableGuard ena;
    void *ptr = malloc(size);
    if (ena) {
        global.on(AllocOp::NewArray, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void *operator new(size_t size, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = malloc(size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, kNone, __builtin_return_address(0));
    }
    return ptr;
}

void *operator new[](size_t size, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = malloc(size);
    if (ena) {
        global.on(AllocOp::NewArray, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    return ptr;
}

#if (__cplusplus >= 201402L || _MSC_VER >= 1916)
void operator delete(void *ptr, size_t size) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr, size_t size) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    free(ptr);
}
#endif

#if (__cplusplus > 201402L || defined(__cpp_aligned_new))
void operator delete(void *ptr, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete(void *ptr, size_t size, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr, size_t size,
                       std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    free(ptr);
}

void *operator new(size_t size, std::align_val_t align) noexcept(false) {
    EnableGuard ena;
    void *ptr = aligned_alloc((size_t)align, size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void *operator new[](size_t size, std::align_val_t align) noexcept(false) {
    EnableGuard ena;
    void *ptr = aligned_alloc((size_t)align, size);
    if (ena) {
        global.on(AllocOp::NewArray, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void *operator new(size_t size, std::align_val_t align,
                   std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = aligned_alloc((size_t)align, size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    return ptr;
}

void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = aligned_alloc((size_t)align, size);
    if (ena) {
        global.on(AllocOp::NewArray, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    return ptr;
}
#endif
