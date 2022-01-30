#pragma once

#include <string>
#include <array>

template <class T, size_t N>
struct _impl_writevdb {
    std::string const &path;
    uint32_t sizex, sizey, sizez;
    std::array<T, N> const *base;

    void operator()() const;
};

template <class T, size_t N = 1>
static void writevdb(std::string const &path, uint32_t sizex, uint32_t sizey, uint32_t sizez, void const *base) {
    _impl_writevdb<T, N>{path, sizex, sizey, sizez, (std::array<T, N> const *)base}();
}

#ifdef WRITEVDB_IMPLEMENTATION
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

namespace {

template <class T, size_t N>
struct vdbtraits {
};

template <>
struct vdbtraits<float, 1> {
    using grid_type = openvdb::FloatGrid;

    template <class T>
    static typename grid_type::ValueType convert(T const &val) {
        return {val[0]};
    }
};

template <>
struct vdbtraits<float, 3> {
    using grid_type = openvdb::Vec3fGrid;

    template <class T>
    static typename grid_type::ValueType convert(T const &val) {
        return {val[0], val[1], val[2]};
    }
};

}

template <class T, size_t N>
void _impl_writevdb<T, N>::operator()() const {
    using GridT = typename vdbtraits<T, N>::grid_type;
    openvdb::tools::Dense<typename GridT::ValueType> dens(openvdb::Coord(sizex, sizey, sizez));
    for (uint32_t z = 0; z < sizez; z++) {
        for (uint32_t y = 0; y < sizey; y++) {
            for (uint32_t x = 0; x < sizex; x++) {
                dens.setValue(x, y, z, vdbtraits<T, N>::convert(base[x + sizex * (y + sizey * z)]));
            }
        }
    }
    auto grid = GridT::create();
    typename GridT::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);
    openvdb::io::File(path).write({grid});
}

template struct _impl_writevdb<float, 1>;
template struct _impl_writevdb<float, 3>;
#endif
