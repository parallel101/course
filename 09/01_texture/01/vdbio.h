#pragma once

#include <string>
#include <functional>
#include <array>

template <class ValT>
struct _impl_writevdb {
    std::string const &path;
    uint32_t sizex, sizey, sizez;
    std::function<void(ValT *, uint32_t, uint32_t)> sampler;

    void operator()() const;
};

template <class ValT, class FuncT>
void writevdb(std::string const &path, uint32_t sizex, uint32_t sizey, uint32_t sizez, FuncT const &func) {
    _impl_writevdb<ValT>{path, sizex, sizey, sizez, [sizex, &func] (ValT *tmp, uint32_t y, uint32_t z) {
            for (uint32_t x = 0; x < sizex; x++) {
                tmp[x] = func(x, y, z);
            }
    }}();
}

#ifdef VDBIO_IMPLEMENTATION
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include "vdbio.h"

namespace {

template <class T>
struct vdbtraits {
    static T convert(T const &val) {
        return val;
    }
};

struct vdbtraits<float> {
    using grid_type = openvdb::FloatGrid;
};

struct vdbtraits<std::array<float, 3>> {
    using grid_type = openvdb::Vec3fGrid;

    template <class T>
    static typename grid_type::ValueType convert(T const &val) {
        return {val[0], val[1], val[2]};
    }
};

}

template <class ValT>
void _impl_writevdb<ValT>::operator()() const {
    auto dummy = [] {
        if constexpr (std::is_same_v<ValT, std::array<float, 3>>)
            return std::decay<openvdb::Vec3fGrid>{};
        else
            return std::decay<openvdb::FloatGrid>{};
    }();
    using GridT = typename vdbtraits<T>::grid_type;
    openvdb::tools::Dense<typename GridT::ValueType> dens(openvdb::Coord(sizex, sizey, sizez));
    std::vector<ValT> tmp(sizex);
    for (uint32_t z = 0; z < sizez; z++) {
        for (uint32_t y = 0; y < sizey; y++) {
            sampler(tmp.data(), y, z);
            for (uint32_t x = 0; x < sizex; x++) {
                dens.setValue(x, y, z, vdbtraits<T>::convert(tmp[x]));
            }
        }
    }
    auto grid = GridT::create();
    typename GridT::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);
    openvdb::io::File(path).write({grid});
}

template struct _impl_writevdb<float>;
template struct _impl_writevdb<std::array<float, 3>>;
#endif
