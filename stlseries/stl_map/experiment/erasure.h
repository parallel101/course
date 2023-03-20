#pragma once

#include <tuple>
#include <utility>
#include <type_traits>
#include <memory>
#include <new>
#include <algorithm>
#include <variant>
#include <typeinfo>

namespace erasure_details {

template <class T, class ...Ts>
struct find_contains_in : std::false_type {
};

template <class T, class T0, class ...Ts>
struct find_contains_in<T, T0, Ts...> : find_contains_in<T, Ts...> {
};

template <class T, class ...Ts>
struct find_contains_in<T, T, Ts...> : std::true_type {
};

template <class T, class ...Ts>
struct find_index_in {
};

template <class T, class T0, class ...Ts>
struct find_index_in<T, T0, Ts...> {
    static constexpr std::size_t value = 1 + find_index_in<T, Ts...>::value;
};

template <class T, class ...Ts>
struct find_index_in<T, T, Ts...> {
    static constexpr std::size_t value = 0;
};

class bad_erasure_cast : public std::bad_cast
{
public:
    virtual const char* what() const noexcept { return "bad erasure::cast"; }
};

template <class ...Tags>
class erasure {
    struct MetaData {
        void (*destruct)(void *);
        /* void (*copy_construct)(void *, void *); */
        std::tuple<
            typename Tags::result(*)(void *, Tags &&)...
            > methods;
        std::type_info const &type_id;
    };
    void *this_ptr;
    MetaData *meta_ptr;

    template <class Self, class ...Args>
    void do_emplace(Args &&...args) {
        constexpr std::size_t meta_offset = std::max(
            sizeof(Self), alignof(MetaData));
        constexpr std::size_t total_alignment = std::max(
            alignof(Self), alignof(MetaData));
        constexpr std::size_t total_size = meta_offset + sizeof(MetaData);
        void *ptr = ::operator new(total_size,
                                   std::align_val_t{total_alignment});
        if constexpr (!std::is_nothrow_constructible_v<Self, Args...>) {
            struct raii_guard {
                void *ptr;

                ~raii_guard() {
                    if (ptr) {
                        ::operator delete(ptr);
                    }
                }
            } guard{ptr};
            this_ptr = new (ptr) Self{std::forward<Args>(args)...};
            guard.ptr = nullptr;
        } else {
            this_ptr = new (ptr) Self{std::forward<Args>(args)...};
        }
        meta_ptr = new (reinterpret_cast<char *>(ptr) + meta_offset) MetaData{
        [] (void *p) {
            reinterpret_cast<Self *>(p)->~Self();
        },
        {
            [] (void *p, Tags &&args) -> typename Tags::result {
                return call(
                    *reinterpret_cast<Self *>(p),
                    std::move(args));
            }...
        }, typeid(Self)};
    }

    template <class Tag>
    typename Tag::result do_call(Tag &&tag) const {
        return std::get<find_index_in<Tag, Tags...>::value>(meta_ptr->methods)
            (this_ptr, std::forward<Tag>(tag));
    }

    void do_destruct() const {
        return meta_ptr->destruct(this_ptr);
    }

public:
    template <class Self, class ...Args, class = std::enable_if_t<
        std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>>>
    explicit erasure(std::in_place_type_t<Self>, Args &&...args) {
        do_emplace<Self>(std::forward<Args>(args)...);
    }

    template <class Self, class = std::enable_if_t<
        std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>>>
    erasure(Self &&self) : erasure{
        std::in_place_type<Self>, std::forward<Self>(self)} {
    }

    template <class Self, class = std::enable_if_t<
        std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>>>
    erasure &operator=(Self &&self) {
        reset();
        do_emplace<Self>(std::move(self));
    }

    template <class Self, class ...Args, class = std::enable_if_t<
        std::is_same_v<Self, std::remove_cv_t<std::remove_reference_t<Self>>>>>
    Self &emplace(Args &&...args) {
        reset();
        do_emplace<Self>(std::forward<Args>(args)...);
        return *reinterpret_cast<Self *>(this_ptr);
    }

    template <class Tag>
    friend typename Tag::result call(
        erasure const &self,
        Tag &&tag) {
        return self.do_call(std::forward<Tag>(tag));
    }

    bool empty() const noexcept {
        return this_ptr != nullptr;
    }

    void reset() noexcept {
        do_destruct();
        ::operator delete(this_ptr);
        this_ptr = nullptr;
        meta_ptr = nullptr;
    }

    std::size_t size() noexcept {
        return reinterpret_cast<char *>(meta_ptr)
            - reinterpret_cast<char *>(this_ptr);
    }

    std::type_info const &type() const noexcept {
        return meta_ptr->type_id;
    }

    template <class T>
    T &cast() const {
        if (type() != typeid(T)) {
            throw bad_erasure_cast{};
        }
        return *reinterpret_cast<T *>(this_ptr);
    }

    template <class T>
    T &cast_unsafe() const noexcept {
        return *reinterpret_cast<T *>(this_ptr);
    }

    erasure(erasure const &from) = delete;
    erasure &operator=(erasure const &from) = delete;

    erasure(erasure &&from) noexcept
        : this_ptr(std::exchange(from.this_ptr, nullptr))
        , meta_ptr(std::exchange(from.meta_ptr, nullptr))
    {}

    erasure &operator=(erasure &&from) noexcept {
        this_ptr = std::exchange(from.this_ptr, nullptr);
        meta_ptr = std::exchange(from.meta_ptr, nullptr);
        return *this;
    }

    void swap(erasure &from) noexcept {
        std::swap(from.this_ptr, this_ptr);
        std::swap(from.meta_ptr, meta_ptr);
    }

    ~erasure() noexcept {
        reset();
    }
};

}

using erasure_details::bad_erasure_cast;
using erasure_details::erasure;
