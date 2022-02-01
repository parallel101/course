#pragma once

struct ctor_t {
};

static constexpr ctor_t ctor;

struct nocopy_t {
    nocopy_t() = default;
    nocopy_t(nocopy_t const &) = delete;
    nocopy_t &operator=(nocopy_t const &) = delete;
    nocopy_t(nocopy_t &&) = delete;
    nocopy_t &operator=(nocopy_t &&) = delete;
};
