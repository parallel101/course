#include <type_traits>
#include <functional>
#include <utility>
#include <string>
#include <map>
#include <iostream>

template
< class Map
, class Iterator = std::conditional_t
    < std::is_const_v<Map>
    , typename Map::const_iterator
    , typename Map::iterator
    >
>
class MapEntry {
    using map_type = std::remove_cv_t<Map>;
    using key_type = typename map_type::key_type;
    using mapped_type = std::conditional_t
        < std::is_const_v<map_type>
        , std::add_const_t<typename map_type::mapped_type>
        , typename map_type::mapped_type
        >;
    using iterator = Iterator;

    map_type &m_map;
    key_type const &m_key;

public:
    explicit MapEntry
        ( map_type &map
        , key_type const &key
        )
        : m_map(map)
        , m_key(key)
    {
    }

    template <class ...Ts>
    mapped_type &or_insert(Ts &&...ts) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return it->second;
        else
            return m_map.try_emplace(it, m_key, std::forward<Ts>(ts)...)->second;
    }

    template <class ...Fs>
    mapped_type &or_insert_with(Fs &&...fs) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return *this;
        else
            return m_map.try_emplace(it, m_key, std::invoke(std::forward<Fs>(fs)...))->second;
    }

    template <class ...Ts>
    mapped_type or_value(Ts &&...ts) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return it->second;
        else
            return mapped_type(std::forward<Ts>(ts)...);
    }

    template <class ...Fs>
    mapped_type or_value_with(Fs &&...fs) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return it->second;
        else
            return std::invoke(std::forward<Fs>(fs)...);
    }

    template <class = void>
    mapped_type &or_die() const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return it->second;
        else
            throw std::out_of_range("MapEntry");
    }

    template <class ...Ts>
    void and_erase(Ts &&...ts) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            return m_map.erase(it);
    }

    template <class ...Fs>
    void and_modify(Fs &&...fs) const {
        iterator it = m_map.find(m_key);
        if (it != m_map.end())
            std::invoke(std::forward<Fs>(fs)..., it->second);
    }

    template <class = void>
    bool exists() const {
        iterator it = m_map.find(m_key);
        return it != m_map.end();
    }
};

template <class Map>
MapEntry(Map, typename Map::key_type const &) -> MapEntry<Map>;

int main() {
    std::map<std::string, int> m;
    MapEntry(m, "hello").or_insert(1);
    std::cout << MapEntry(m, "hello").or_value(3) << std::endl;
    std::cout << MapEntry(m, "world").or_value(3) << std::endl;
    MapEntry(m, "hello").and_modify([&] (auto &val) { val += 1; });
    std::cout << MapEntry(m, "world").or_insert(4) << std::endl;
    return 0;
}
