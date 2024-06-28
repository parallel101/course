#pragma once

#include <string>
#include <memory>
#include <sstream>
#include <utility>
#include <json/json.h>
#include "reflect.hpp"

namespace reflect_json {

inline std::string jsonToStr(Json::Value root) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ostringstream os;
    writer->write(root, &os);
    return os.str();
}

inline Json::Value strToJson(std::string const &json) {
    Json::Value root;
    Json::Reader reader;
    reader.parse(json, root);
    return root;
}

template <class T>
struct special_traits {
    static constexpr bool value = false;
};

template <class T, std::enable_if_t<!reflect::has_member<T>() && !special_traits<T>::value, int> = 0>
Json::Value objToJson(T const &object) {
    return object;
}

template <class T, std::enable_if_t<!reflect::has_member<T>() && special_traits<T>::value, int> = 0>
Json::Value objToJson(T const &object) {
    return special_traits<T>::objToJson(object);
}

template <class T, std::enable_if_t<reflect::has_member<T>(), int> = 0>
Json::Value objToJson(T const &object) {
    Json::Value root;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        root[key] = objToJson(value);
    });
    return root;
}

template <class T, std::enable_if_t<!reflect::has_member<T>() && !special_traits<T>::value, int> = 0>
T jsonToObj(Json::Value const &root) {
    return root.as<T>();
}

template <class T, std::enable_if_t<!reflect::has_member<T>() && special_traits<T>::value, int> = 0>
T jsonToObj(Json::Value const &root) {
    return special_traits<T>::jsonToObj(root);
}

template <class T, std::enable_if_t<reflect::has_member<T>(), int> = 0>
T jsonToObj(Json::Value const &root) {
    T object;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        value = jsonToObj<std::decay_t<decltype(value)>>(root[key]);
    });
    return object;
}

template <class T, class Alloc>
struct special_traits<std::vector<T, Alloc>> {
    static constexpr bool value = true;

    static Json::Value objToJson(std::vector<T, Alloc> const &object) {
        Json::Value root;
        for (auto const &elem: object) {
            root.append(reflect_json::objToJson(elem));
        }
        return root;
    }

    static std::vector<T, Alloc> jsonToObj(Json::Value const &root) {
        std::vector<T, Alloc> object;
        for (auto const &elem: root) {
            object.push_back(reflect_json::jsonToObj<T>(elem));
        }
        return object;
    }
};

template <class K, class V, class Alloc>
struct special_traits<std::map<K, V, Alloc>> {
    static constexpr bool value = true;

    static Json::Value objToJson(std::map<K, V, Alloc> const &object) {
        Json::Value root;
        for (auto const &elem: object) {
            root[elem.first] = reflect_json::objToJson(elem.second);
        }
        return root;
    }

    static std::map<K, V, Alloc> jsonToObj(Json::Value const &root) {
        std::map<K, V, Alloc> object;
        for (auto const &key: root.getMemberNames()) {
            object[key] = reflect_json::jsonToObj<V>(root[key]);
        }
        return object;
    }
};

template <class T>
std::string serialize(T const &object) {
    return jsonToStr(objToJson(object));
}

template <class T>
T deserialize(std::string const &json) {
    return jsonToObj<T>(strToJson(json));
}

}
