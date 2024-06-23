#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace pengsig {

enum class CallbackResult {
    Keep,
    Erase,
};

constexpr struct oneshot_t {
    explicit oneshot_t() = default;
} oneshot;

enum class nshot_t : size_t {};

namespace details_ {

    template <class Self>
    std::shared_ptr<Self> lock_if_weak(std::weak_ptr<Self> const &self) {
        return self.lock();
    }

    template <class Self>
    Self const &lock_if_weak(Self const &self) {
        return self;
    }

    template <class Self, class MemFn>
    auto bind(Self self, MemFn memfn, oneshot_t) {
        return [self = std::move(self), memfn] (auto ...t) { // define t
            auto const &ptr = details_::lock_if_weak(self);
            if (ptr == nullptr) {
                return CallbackResult::Erase;
            }
            ((*self).*memfn)(t...); // use t
            return CallbackResult::Erase;
        };
    }

    template <class Self, class MemFn>
    auto bind(Self self, MemFn memfn, nshot_t n) {
        return [self = std::move(self), memfn, n = static_cast<size_t>(n)] (auto ...t) mutable { // define t
            if (n == 0) {
                return CallbackResult::Erase;
            }
            auto const &ptr = details_::lock_if_weak(self);
            if (ptr == nullptr) {
                return CallbackResult::Erase;
            }
            ((*ptr).*memfn)(t...); // use t
            --n;
            if (n == 0) {
                return CallbackResult::Erase;
            }
            return CallbackResult::Keep;
        };
    }

    template <class Self, class MemFn>
    auto bind(Self self, MemFn memfn) {
        return [self = std::move(self), memfn] (auto ...t) { // define t
            auto const &ptr = details_::lock_if_weak(self);
            if (ptr == nullptr) {
                return CallbackResult::Erase;
            }
            ((*ptr).*memfn)(t...); // use t
            return CallbackResult::Keep;
        };
    }

}

// 形参包语法
// ...T 省略号在左边表示 define，定义一个形参包
// T... 省略号在右边表示 use，使用一个形参包

template <class ...T> // define T
struct Signal {
private:
#if __cpp_lib_move_only_function // standard feature-test macro
    using Functor = std::move_only_function<void(T...)>; // use T
#else
    using Functor = std::function<CallbackResult(T...)>; // use T
#endif

    std::vector<Functor> m_callbacks;

public:
#if __cpp_if_constexpr
    template <class Func>
    void connect(Func callback) {
        if constexpr (std::is_invocable_r_v<CallbackResult, Func, T...>) {
            m_callbacks.push_back(std::move(callback));
        } else {
            m_callbacks.push_back([callback = std::move(callback)] (T ...t) mutable { // define t
                callback(std::forward<T>(t)...); // use t
                return CallbackResult::Keep;
            });
        }
    }
#else
    template <class Func, typename std::enable_if<std::is_convertible<decltype(std::declval<Func>()(std::declval<T>()...)), CallbackResult>::value, int>::type = 0>
    void connect(Func callback) {
        m_callbacks.push_back(std::move(callback));
    }

    template <class Func, typename std::enable_if<std::is_void<decltype(std::declval<Func>()(std::declval<T>()...))>::value, int>::type = 0>
    void connect(Func callback) {
        m_callbacks.push_back([callback = std::move(callback)] (T ...t) mutable { // define t
            callback(std::forward<T>(t)...); // use t
            return CallbackResult::Keep;
        });
    }
#endif

    template <class Self, class MemFn, class ...Tag> // define Tag
    void connect(Self self, MemFn memfn, Tag ...tag) { // use Tag, define tag
        m_callbacks.push_back(details_::bind(std::move(self), memfn, tag...)); // use tag
    }

    void emit(T ...t) { // use T, define t
        for (auto it = m_callbacks.begin(); it != m_callbacks.end();) {
            CallbackResult res = (*it)(t...);
            switch (res) {
            case CallbackResult::Keep:
                ++it;
                break;
            case CallbackResult::Erase:
                it = m_callbacks.erase(it);
                break;
            };
        }
    }
};

#if __cplusplus >= 202002L && !(defined(_MSC_VER) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL))
#define PENGSIG_FUN(_fun, ...) [=] (auto &&...__t) { return _fun(__VA_ARGS__ __VA_OPT__(,) std::forward<decltype(_t)>(_t)...); }
#else
#define PENGSIG_FUN(_fun) [=] (auto &&..._t) { return __fun(std::forward<decltype(_t)>(_t)...); }
#endif

}
