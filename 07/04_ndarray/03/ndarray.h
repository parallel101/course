#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <type_traits>

template <std::size_t N, class T>
class ndarray {
    static_assert(N > 0);
    static_assert(std::is_same_v<std::remove_reference_t<std::remove_cv_t<T>>, T>);

    using Dim = std::array<std::size_t, N>;

    std::vector<T> m_arr;
    Dim m_shape{};

    constexpr static std::size_t _calc_size(Dim const &dim) noexcept {
        std::size_t size = dim[0];
        for (std::size_t i = 1; i < N; i++)
            size *= dim[i];
        return size;
    }

public:
    ndarray() = default;
    ndarray(ndarray const &) = default;
    ndarray(ndarray &&) = default;
    ndarray &operator=(ndarray const &) = default;
    ndarray &operator=(ndarray &&) = default;
    ~ndarray() = default;

    explicit ndarray(Dim const &dim)
        : m_arr(_calc_size(dim))
        , m_shape(dim)
    {
    }

    explicit ndarray(Dim const &dim, T const &t)
        : m_arr(_calc_size(dim))
        , m_shape(dim)
    {
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    explicit ndarray(Ts const &...ts)
        : ndarray(Dim{ts...})
    {
    }

    void reshape(Dim const &dim)
    {
        std::size_t size = _calc_size(dim);
        m_shape = dim;
        m_arr.clear();
        m_arr.resize(size);
    }

    void reshape(Dim const &dim, T const &t)
    {
        std::size_t size = _calc_size(dim);
        m_shape = dim;
        m_arr.clear();
        m_arr.resize(size);
    }

    void shrink_to_fit()
    {
        m_arr.shrink_to_fit();
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    void reshape(Ts const &...ts)
    {
        this->reshape(Dim{ts...});
    }

    constexpr Dim shape() const noexcept
    {
        return m_shape;
    }

    constexpr Dim shape(std::size_t i) const noexcept
    {
        return m_shape[i];
    }

    constexpr std::size_t linearize(Dim const &dim) const noexcept
    {
        std::size_t offset = dim[0];
        std::size_t term = 1;
        for (std::size_t i = 1; i < N; i++) {
            term *= m_shape[i - 1];
            offset += term * dim[i];
        }
        return offset;
    }

    std::size_t safe_linearize(Dim const &dim) const
    {
        for (std::size_t i = 0; i < N; i++) {
            if (dim[i] > m_shape[i])
                throw std::out_of_range("ndarray::at");
        }
        return linearize(dim);
    }

    T &operator()(Dim const &dim) noexcept
    {
        return m_arr[linearize(dim)];
    }

    T const &operator()(Dim const &dim) const noexcept
    {
        return m_arr[linearize(dim)];
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    T &operator()(Ts const &...ts) noexcept
    {
        return operator()(Dim{ts...});
    }

    template <class ...Ts, std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), int> = 0>
    T const &operator()(Ts const &...ts) const noexcept
    {
        return operator()(Dim{ts...});
    }

    T &operator[](Dim const &dim) noexcept
    {
        return operator()(linearize(dim));
    }

    T const &operator[](Dim const &dim) const noexcept
    {
        return operator()(linearize(dim));
    }

    T &at(Dim const &dim)
    {
        return m_arr[safe_linearize(dim)];
    }

    T const &at(Dim const &dim) const
    {
        return m_arr[safe_linearize(dim)];
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
