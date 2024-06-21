#include <cstdint>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <string>

extern "C" void *__libc_malloc(size_t size);
extern "C" void __libc_free(void *ptr);

inline std::string addr2sym(void *addr) {
    std::string result;
    char **strings = backtrace_symbols(&addr, 1);
    if (strings) {
        result = strings[0];
    }
    free(strings);
    return result;
}

struct AddrOp {
    void *ptr;
    size_t size;
    void *caller;
    uint64_t timestamp;
    uint32_t tid;
    enum : uint8_t {
        type_malloc,
        type_free,
        type_new,
        type_delete,
        type_new_array,
        type_delete_array,
    } type;
    uint8_t p2align;
};

struct GlobalData {
    GlobalData() {
        enable = true;
    }

    ~GlobalData() {
        enable = false;
        for (auto &&[ptr, caller] : allocated) {
            printf("检测到内存泄漏，地址: %p，调用者: %s\n", ptr, addr2sym(caller).c_str());
        }
    }

    std::map<void *, void *> allocated;
    bool enable = false;

    void on_malloc(void *ptr, size_t size, void *caller) {
        if (enable && ptr) {
            enable = false;
            printf("%p -> malloc(%zu) = %p\n", caller, size, ptr);
            allocated.insert({ptr, caller});
            enable = true;
        }
    }

    void on_free(void *ptr, void *caller) {
        if (enable && ptr) {
            enable = false;
            printf("%p -> free(%p)\n", caller, ptr);
            if (!allocated.erase(ptr)) {
                printf("尝试释放无效指针: %p，调用者: %s\n", ptr, addr2sym(caller).c_str());
            }
            enable = true;
        }
    }
} hook;

extern "C" void *malloc(size_t size) {
    void *ptr = __libc_malloc(size);
    hook.on_malloc(ptr, size, __builtin_return_address(0));
    return ptr;
}

extern "C" void free(void *ptr) {
    hook.on_free(ptr, __builtin_return_address(0));
    __libc_free(ptr);
}

int main() {
    void *p1 = malloc(32);
    void *p2 = malloc(64);
    free(p1);
}
