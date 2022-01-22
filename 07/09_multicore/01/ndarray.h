#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include "alignalloc.h"

template <std::size_t N, class T, std::size_t LoBound = 0, std::size_t HiBound = LoBound, class AllocatorT = AlignedAllocator<T>>
class ndarray {
    static_assert(N > 0, "N cannot be 0");
    static_assert(std::is_same_v<std::remove_reference_t<std::remove_cv_t<T>>, T>, "T cannot be cvref");

    using Dim = std::array<std::intptr_t, N>;
    using Shape = std::array<std::size_t, N>;

    std::vector<T, AllocatorT> m_arr;
    Shape m_shape{};

    constexpr static std::size_t _calc_size(Shape const &shape) noexcept
    {
        std::size_t size = shape[0] + (LoBound + HiBound);
        for (std::size_t i = 1; i < N; i++) {
            size *= shape[i] + (LoBound + HiBound);
        }
        return size;
    }

public:
    ndarray() = default;
    ndarray(ndarray const &) = default;
    ndarray(ndarray &&) = default;
    ndarray &operator=(ndarray const &) = default;
    ndarray &operator=(ndarray &&) = default;
    ~ndarray() = default;

    explicit ndarray(Shape const &shape)
        : m_arr(_calc_size(shape))
        , m_shape(shape)
    {
    }

    explicit ndarray(Shape const &shape, T const &value)
        : m_arr(_calc_size(shape), value)
        , m_shape(shape)
    {
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    explicit ndarray(Ts const &...ts)
        : ndarray(Shape{ts...})
    {
    }

    void reshape(Shape const &shape)
    {
        std::size_t size = _calc_size(shape);
        m_shape = shape;
        m_arr.clear();
        m_arr.resize(size);
    }

    void reshape(Shape const &shape, T const &value)
    {
        std::size_t size = _calc_size(shape);
        m_shape = shape;
        m_arr.clear();
        m_arr.resize(size, value);
    }

    void shrink_to_fit()
    {
        m_arr.shrink_to_fit();
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    void reshape(Ts const &...ts)
    {
        this->reshape(Shape{ts...});
    }

    constexpr Shape shape() const noexcept
    {
        return m_shape;
    }

    constexpr std::size_t shape(std::size_t i) const noexcept
    {
        return m_shape[i];
    }

    constexpr std::size_t linearize(Dim const &dim) const noexcept
    {
        std::size_t offset{dim[0] + LoBound};
        std::size_t term = 1;
        for (std::size_t i = 1; i < N; i++) {
            term *= m_shape[i - 1] + (LoBound + HiBound);
            offset += term * std::size_t{dim[i] + LoBound};
        }
        return offset;
    }

    std::size_t safe_linearize(Dim const &dim) const
    {
        for (std::size_t i = 0; i < N; i++) {
            if (dim[i] < -std::intptr_t{LoBound} || dim[i] >= m_shape[i] + HiBound)
                throw std::out_of_range("ndarray::at");
        }
        return linearize(dim);
    }

    constexpr T *data() noexcept
    {
        return m_arr.data();
    }

    constexpr T const *data() const noexcept
    {
        return m_arr.data();
    }

    constexpr T &operator()(Dim const &dim) noexcept
    {
        return data()[linearize(dim)];
    }

    constexpr T const &operator()(Dim const &dim) const noexcept
    {
        return data()[linearize(dim)];
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    constexpr T &operator()(Ts const &...ts) noexcept
    {
        return operator()(Dim{ts...});
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    constexpr T const &operator()(Ts const &...ts) const noexcept
    {
        return operator()(Dim{ts...});
    }

    constexpr T &operator[](Dim const &dim) noexcept
    {
        return operator()(linearize(dim));
    }

    constexpr T const &operator[](Dim const &dim) const noexcept
    {
        return operator()(linearize(dim));
    }

    T &at(Dim const &dim)
    {
        return data()[safe_linearize(dim)];
    }

    T const &at(Dim const &dim) const
    {
        return data()[safe_linearize(dim)];
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    T &at(Ts const &...ts)
    {
        return at(Dim{ts...});
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    T const &at(Ts const &...ts) const
    {
        return at(Dim{ts...});
    }
};
