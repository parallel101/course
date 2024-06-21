#include <cerrno>
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

#if __GNUC__
extern "C" void *__libc_malloc(size_t size) noexcept;
extern "C" void __libc_free(void *ptr) noexcept;
extern "C" void *__libc_calloc(size_t nmemb, size_t size) noexcept;
extern "C" void *__libc_realloc(void *ptr, size_t size) noexcept;
extern "C" void *__libc_reallocarray(void *ptr, size_t nmemb,
                                     size_t size) noexcept;
extern "C" void *__libc_valloc(size_t size) noexcept;
extern "C" void *__libc_memalign(size_t align, size_t size) noexcept;
#define REAL_LIBC(name) __libc_##name
#define MAY_OVERRIDE_MALLOC 1
#define MAY_SUPPORT_MEMALIGN 1

#elif _MSC_VER
static void *msvc_malloc(size_t size) noexcept {
    return HeapAlloc(GetProcessHeap(), 0, size);
}

static void *msvc_calloc(size_t nmemb, size_t size) noexcept {
    return HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, nmemb * size);
}

static void msvc_free(void *ptr) noexcept {
    HeapFree(GetProcessHeap(), 0, ptr);
}

static void *msvc_realloc(void *ptr, size_t size) noexcept {
    return HeapReAlloc(GetProcessHeap(), 0, ptr, size);
}

static void *msvc_reallocarray(void *ptr, size_t nmemb,
                               size_t size) noexcept {
    return msvc_realloc(ptr, nmemb * size);
}

#define REAL_LIBC(name) msvc_##name
#define MAY_OVERRIDE_MALLOC 1
#define MAY_SUPPORT_MEMALIGN 0

#else
#define REAL_LIBC(name) name
#define MAY_OVERRIDE_MALLOC 0
#define MAY_SUPPORT_MEMALIGN 0
#endif

#if MAY_OVERRIDE_MALLOC
extern "C" void *malloc(size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(malloc)(size);
    if (ena) {
        global.on(AllocOp::Malloc, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    return ptr;
}

extern "C" void free(void *ptr) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Free, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

extern "C" void *calloc(size_t nmemb, size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(calloc)(nmemb, size);
    if (ena) {
        global.on(AllocOp::Malloc, ptr, nmemb * size, kNone,
                  __builtin_return_address(0));
    }
    return ptr;
}

extern "C" void *realloc(void *ptr, size_t size) noexcept {
    EnableGuard ena;
    void *new_ptr = REAL_LIBC(realloc)(ptr, size);
    if (ena) {
        global.on(AllocOp::Malloc, new_ptr, size, kNone,
                  __builtin_return_address(0));
        if (new_ptr) {
            global.on(AllocOp::Free, ptr, kNone, kNone,
                      __builtin_return_address(0));
        }
    }
    return new_ptr;
}

extern "C" void *reallocarray(void *ptr, size_t nmemb, size_t size) noexcept {
    EnableGuard ena;
    void *new_ptr = REAL_LIBC(reallocarray)(ptr, nmemb, size);
    if (ena) {
        global.on(AllocOp::Malloc, new_ptr, nmemb * size, kNone,
                  __builtin_return_address(0));
        if (new_ptr) {
            global.on(AllocOp::Free, ptr, kNone, kNone,
                      __builtin_return_address(0));
        }
    }
    return new_ptr;
}

#if MAY_SUPPORT_MEMALIGN
extern "C" void *valloc(size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(valloc)(size);
    if (ena) {
#if __unix__
        size_t pagesize = sysconf(_SC_PAGESIZE);
#elif _WIN32
        SYSTEM_INFO info;
        info.dwPageSize = kNone;
        GetSystemInfo(&info);
        size_t pagesize = info.dwPageSize;
#else
        size_t pagesize = 0;
#endif
        global.on(AllocOp::Malloc, ptr, size, pagesize,
                  __builtin_return_address(0));
    }
    return ptr;
}

extern "C" void *memalign(size_t align, size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(memalign)(align, size);
    if (ena) {
        global.on(AllocOp::Malloc, ptr, size, align,
                  __builtin_return_address(0));
    }
    return ptr;
}

extern "C" void *aligned_alloc(size_t align, size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(memalign)(align, size);
    if (ena) {
        global.on(AllocOp::Malloc, ptr, size, align,
                  __builtin_return_address(0));
    }
    return ptr;
}

extern "C" int posix_memalign(void **memptr, size_t align, size_t size) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(memalign)(align, size);
    if (ena) {
        global.on(AllocOp::Malloc, *memptr, size, align,
                  __builtin_return_address(0));
    }
    int ret = 0;
    if (!ptr) {
        ret = errno;
    } else {
        *memptr = ptr;
    }
    return ret;
}
#endif
#endif

void operator delete(void *ptr) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void *operator new(size_t size) noexcept(false) {
    EnableGuard ena;
    void *ptr = REAL_LIBC(malloc)(size);
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
    void *ptr = REAL_LIBC(malloc)(size);
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
    void *ptr = REAL_LIBC(malloc)(size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, kNone, __builtin_return_address(0));
    }
    return ptr;
}

void *operator new[](size_t size, std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(malloc)(size);
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
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, size_t size) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, size, kNone,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}
#endif

#if (__cplusplus > 201402L || defined(__cpp_aligned_new))
#if MAY_SUPPORT_MEMALIGN
void operator delete(void *ptr, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, size_t size, std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, size_t size,
                       std::align_val_t align) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete(void *ptr, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::Delete, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void operator delete[](void *ptr, std::align_val_t align,
                       std::nothrow_t const &) noexcept {
    EnableGuard ena;
    if (ena) {
        global.on(AllocOp::DeleteArray, ptr, kNone, (size_t)align,
                  __builtin_return_address(0));
    }
    REAL_LIBC(free)(ptr);
}

void *operator new(size_t size, std::align_val_t align) noexcept(false) {
    EnableGuard ena;
    void *ptr = REAL_LIBC(memalign)((size_t)align, size);
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
    void *ptr = REAL_LIBC(memalign)((size_t)align, size);
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
    void *ptr = REAL_LIBC(memalign)((size_t)align, size);
    if (ena) {
        global.on(AllocOp::New, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    return ptr;
}

void *operator new[](size_t size, std::align_val_t align,
                     std::nothrow_t const &) noexcept {
    EnableGuard ena;
    void *ptr = REAL_LIBC(memalign)((size_t)align, size);
    if (ena) {
        global.on(AllocOp::NewArray, ptr, size, (size_t)align,
                  __builtin_return_address(0));
    }
    return ptr;
}
#endif
#endif
