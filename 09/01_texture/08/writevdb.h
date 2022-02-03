#pragma once

#include <string>
#include <array>
#include <vector>
#include <memory>

class VDBWriter {
    struct Impl;
    std::unique_ptr<Impl> const impl;

    template <class T, size_t N>
    struct AddGridImpl {
        VDBWriter *that;
        std::string name;
        void const *base;
        uint32_t sizex, sizey, sizez;
        int32_t minx, miny, minz;
        uint32_t pitchx, pitchy, pitchz;

        void operator()() const;
    };

public:
    VDBWriter();
    ~VDBWriter();

    VDBWriter(VDBWriter const &) = delete;
    VDBWriter &operator=(VDBWriter const &) = delete;
    VDBWriter(VDBWriter &&) = delete;
    VDBWriter &operator=(VDBWriter &&) = delete;

    template <class T, size_t N, bool normalizedCoords = true>
    void addGrid(std::string const &name, void const *base, uint32_t sizex, uint32_t sizey, uint32_t sizez, uint32_t pitchx = 0, uint32_t pitchy = 0, uint32_t pitchz = 0) {
        if (pitchx == 0) pitchx = sizeof(T) * N;
        if (pitchy == 0) pitchy = pitchx * sizex;
        if (pitchz == 0) pitchz = pitchy * sizey;
        int32_t minx = normalizedCoords ? -(int32_t)sizex / 2 : 0;
        int32_t miny = normalizedCoords ? -(int32_t)sizey / 2 : 0;
        int32_t minz = normalizedCoords ? -(int32_t)sizez / 2 : 0;
        AddGridImpl<T, N>{this, name, base, sizex, sizey, sizez, minx, miny, minz, pitchx, pitchy, pitchz}();
    }

    void write(std::string const &path);
};

#ifdef WRITEVDB_IMPLEMENTATION
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

struct VDBWriter::Impl {
    openvdb::GridPtrVec grids;
};

VDBWriter::VDBWriter() : impl(std::make_unique<Impl>()) {
}

VDBWriter::~VDBWriter() = default;

void VDBWriter::write(std::string const &path) {
    openvdb::io::File(path).write(impl->grids);
}

namespace {

template <class T, size_t N>
struct vdbtraits {
};

template <>
struct vdbtraits<float, 1> {
    using type = openvdb::FloatGrid;
};

template <>
struct vdbtraits<float, 3> {
    using type = openvdb::Vec3fGrid;
};


template <class VecT, class T, size_t ...Is>
VecT help_make_vec(T const *ptr, std::index_sequence<Is...>) {
    return VecT(ptr[Is]...);
}

}

template <class T, size_t N>
void VDBWriter::AddGridImpl<T, N>::operator()() const {
    using GridT = typename vdbtraits<T, N>::type;

    openvdb::tools::Dense<typename GridT::ValueType> dens(openvdb::Coord(sizex, sizey, sizez), openvdb::Coord(minx, miny, minz));
    for (uint32_t z = 0; z < sizez; z++) {
        for (uint32_t y = 0; y < sizey; y++) {
            for (uint32_t x = 0; x < sizex; x++) {
                auto ptr = reinterpret_cast<T const *>(reinterpret_cast<char const *>(base) + pitchx * x + pitchy * y + pitchz * z);
                dens.setValue(x, y, z, help_make_vec<typename GridT::ValueType>(ptr, std::make_index_sequence<N>{}));
            }
        }
    }

    auto grid = GridT::create();
    typename GridT::ValueType tolerance{0};
    openvdb::tools::copyFromDense(dens, grid->tree(), tolerance);

    openvdb::MetaMap &meta = *grid;
    meta.insertMeta(openvdb::Name("name"), openvdb::TypedMetadata<std::string>(name));
    that->impl->grids.push_back(grid);
}

template struct VDBWriter::AddGridImpl<float, 1>;
template struct VDBWriter::AddGridImpl<float, 3>;
#endif
