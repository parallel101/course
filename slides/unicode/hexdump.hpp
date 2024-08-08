#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>
#include <iostream>
#include <iomanip>

template <std::size_t kChunk = 16, class T>
void hexdump(T const *data, std::size_t size) {
    std::size_t addr = 0;
    std::vector<char> saved;
    for (std::size_t addr = 0; addr != size; ++addr) {
        std::cout << std::setw(8) << std::setfill('0') << std::hex << addr << ' ';
        for (; addr != size; ++addr) {
            T c = data[addr];
            std::cout << ' ' << std::right << std::hex << std::setw(2 * sizeof(T)) << std::setfill('0');
            std::cout << static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<T>>(c));
            if constexpr (sizeof(T) == sizeof(char) && std::is_convertible_v<T, char>) {
                saved.push_back(static_cast<char>(c));
            }
        }
        if constexpr (sizeof(T) == sizeof(char) && std::is_convertible_v<T, char>) {
            if (addr % kChunk != 0) {
                for (std::size_t i = 0; i < (kChunk - addr % kChunk) * 3; i++) {
                    std::cout << ' ';
                }
            }
            std::cout << "  |";
            for (auto c: saved) {
                if (!std::isprint(c)) {
                    c = '.';
                }
                std::cout << c;
            }
            std::cout << "|";
            saved.clear();
        }
        std::cout << '\n';
    }
}

template <std::size_t kChunk = 16, class T>
void hexdump(T const &t) {
#if __cpp_lib_nonmember_container_access
    using std::data;
    using std::size;
    hexdump<kChunk>(data(t), size(t));
#else
    hexdump<kChunk>(t.data(), t.size());
#endif
}
