#pragma once

#include <string>
#include <memory>
#include <sstream>
#include <json/json.h> // jsoncpp
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

template <class T, std::enable_if_t<!reflect::has_member<T>(), int> = 0>
Json::Value objToJson(T const &object) {
    return object;
}

template <class T, std::enable_if_t<reflect::has_member<T>(), int> = 0>
Json::Value objToJson(T const &object) {
    Json::Value root;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        root[key] = objToJson(value);
    });
    return root;
}

template <class T, std::enable_if_t<!reflect::has_member<T>(), int> = 0>
T jsonToObj(Json::Value const &root) {
    return root.as<T>();
}

template <class T, std::enable_if_t<reflect::has_member<T>(), int> = 0>
T jsonToObj(Json::Value const &root) {
    T object;
    reflect::foreach_member(object, [&](const char *key, auto &value) {
        value = jsonToObj<std::decay_t<decltype(value)>>(root[key]);
    });
    return object;
}

template <class T>
std::string serialize(T const &object) {
    return jsonToStr(objToJson(object));
}

template <class T>
T deserialize(std::string const &json) {
    return jsonToObj<T>(strToJson(json));
}

}
