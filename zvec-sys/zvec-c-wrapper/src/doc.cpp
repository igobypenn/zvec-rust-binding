#include "zvec_c_internal.h"
#include <cstring>

namespace {

template<typename T>
zvec_status_t set_field_helper(zvec::Doc* doc, const char* field, T value) {
    if (!doc || !field) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid doc or field");
        return s;
    }
    doc->set(std::string(field), std::move(value));
    return zvec_wrapper::ok_status();
}

}

extern "C" {

zvec_doc_t* zvec_doc_new(void) {
    auto* doc = new zvec_doc_t;
    doc->ptr = std::make_shared<zvec::Doc>();
    doc->owned = true;
    return doc;
}

void zvec_doc_free(zvec_doc_t* doc) {
    if (doc && doc->owned) {
        delete doc;
    }
}

void zvec_doc_set_pk(zvec_doc_t* doc, const char* pk) {
    if (doc && doc->ptr && pk) {
        doc->ptr->set_pk(std::string(pk));
    }
}

const char* zvec_doc_pk(const zvec_doc_t* doc) {
    if (doc && doc->ptr) {
        return doc->ptr->pk().c_str();
    }
    return nullptr;
}

void zvec_doc_set_score(zvec_doc_t* doc, float score) {
    if (doc && doc->ptr) {
        doc->ptr->set_score(score);
    }
}

float zvec_doc_score(const zvec_doc_t* doc) {
    if (doc && doc->ptr) {
        return doc->ptr->score();
    }
    return 0.0f;
}

void zvec_doc_set_doc_id(zvec_doc_t* doc, uint64_t doc_id) {
    if (doc && doc->ptr) {
        doc->ptr->set_doc_id(doc_id);
    }
}

uint64_t zvec_doc_doc_id(const zvec_doc_t* doc) {
    if (doc && doc->ptr) {
        return doc->ptr->doc_id();
    }
    return 0;
}

zvec_status_t zvec_doc_set_bool(zvec_doc_t* doc, const char* field, bool value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_int32(zvec_doc_t* doc, const char* field, int32_t value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_int64(zvec_doc_t* doc, const char* field, int64_t value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_uint32(zvec_doc_t* doc, const char* field, uint32_t value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_uint64(zvec_doc_t* doc, const char* field, uint64_t value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_float(zvec_doc_t* doc, const char* field, float value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_double(zvec_doc_t* doc, const char* field, double value) {
    return set_field_helper(doc->ptr.get(), field, value);
}

zvec_status_t zvec_doc_set_string(zvec_doc_t* doc, const char* field, const char* value) {
    if (!doc || !doc->ptr || !field) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid doc or field");
        return s;
    }
    doc->ptr->set<std::string>(std::string(field), std::string(value));
    return zvec_wrapper::ok_status();
}

void zvec_doc_set_null(zvec_doc_t* doc, const char* field) {
    if (doc && doc->ptr && field) {
        doc->ptr->set_null(std::string(field));
    }
}

zvec_status_t zvec_doc_set_vector_fp32(zvec_doc_t* doc, const char* field, const float* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<float> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_vector_fp64(zvec_doc_t* doc, const char* field, const double* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<double> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_vector_int8(zvec_doc_t* doc, const char* field, const int8_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int8_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_vector_int16(zvec_doc_t* doc, const char* field, const int16_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int16_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_vector_int32(zvec_doc_t* doc, const char* field, const int32_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int32_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_vector_int64(zvec_doc_t* doc, const char* field, const int64_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int64_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_sparse_vector_fp32(zvec_doc_t* doc, const char* field,
    const uint32_t* indices, size_t indices_count, const float* values, size_t values_count) {
    if (!doc || !doc->ptr || !field || !indices || !values || indices_count != values_count) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<uint32_t> idx(indices, indices + indices_count);
    std::vector<float> vals(values, values + values_count);
    doc->ptr->set(std::string(field), std::make_pair(std::move(idx), std::move(vals)));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_array_int32(zvec_doc_t* doc, const char* field, const int32_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int32_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_array_int64(zvec_doc_t* doc, const char* field, const int64_t* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<int64_t> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_array_float(zvec_doc_t* doc, const char* field, const float* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<float> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_array_double(zvec_doc_t* doc, const char* field, const double* data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<double> vec(data, data + len);
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_doc_set_array_string(zvec_doc_t* doc, const char* field, const char** data, size_t len) {
    if (!doc || !doc->ptr || !field || !data) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::vector<std::string> vec;
    vec.reserve(len);
    for (size_t i = 0; i < len; i++) {
        vec.emplace_back(data[i]);
    }
    doc->ptr->set(std::string(field), std::move(vec));
    return zvec_wrapper::ok_status();
}

bool zvec_doc_get_bool(const zvec_doc_t* doc, const char* field, bool* out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<bool>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value();
            return true;
        }
    }
    return false;
}

bool zvec_doc_get_int32(const zvec_doc_t* doc, const char* field, int32_t* out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<int32_t>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value();
            return true;
        }
    }
    return false;
}

bool zvec_doc_get_int64(const zvec_doc_t* doc, const char* field, int64_t* out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<int64_t>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value();
            return true;
        }
    }
    return false;
}

bool zvec_doc_get_float(const zvec_doc_t* doc, const char* field, float* out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<float>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value();
            return true;
        }
    }
    return false;
}

bool zvec_doc_get_double(const zvec_doc_t* doc, const char* field, double* out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<double>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value();
            return true;
        }
    }
    return false;
}

bool zvec_doc_get_string(const zvec_doc_t* doc, const char* field, const char** out_value) {
    if (doc && doc->ptr && field && out_value) {
        auto result = doc->ptr->get<std::string>(std::string(field));
        if (result.has_value()) {
            *out_value = result.value().c_str();
            return true;
        }
    }
    return false;
}

size_t zvec_doc_get_vector_fp32(const zvec_doc_t* doc, const char* field, float* out_data, size_t max_len) {
    if (doc && doc->ptr && field && out_data && max_len > 0) {
        auto result = doc->ptr->get<std::vector<float>>(std::string(field));
        if (result.has_value()) {
            const auto& vec = result.value();
            size_t copy_len = std::min(vec.size(), max_len);
            std::memcpy(out_data, vec.data(), copy_len * sizeof(float));
            return vec.size();
        }
    }
    return 0;
}

bool zvec_doc_has(const zvec_doc_t* doc, const char* field) {
    return doc && doc->ptr && field && doc->ptr->has(std::string(field));
}

bool zvec_doc_has_value(const zvec_doc_t* doc, const char* field) {
    return doc && doc->ptr && field && doc->ptr->has_value(std::string(field));
}

bool zvec_doc_is_null(const zvec_doc_t* doc, const char* field) {
    return doc && doc->ptr && field && doc->ptr->is_null(std::string(field));
}

zvec_string_array_t zvec_doc_field_names(const zvec_doc_t* doc) {
    zvec_string_array_t result = {nullptr, 0};
    if (doc && doc->ptr) {
        auto names = doc->ptr->field_names();
        result.count = names.size();
        result.strings = (char**)malloc(sizeof(char*) * names.size());
        for (size_t i = 0; i < names.size(); i++) {
            result.strings[i] = strdup(names[i].c_str());
        }
    }
    return result;
}

void zvec_doc_list_free(zvec_doc_list_t* list) {
    if (list) {
        for (size_t i = 0; i < list->count; i++) {
            if (list->docs[i]) {
                list->docs[i]->owned = false;
                zvec_doc_free(list->docs[i]);
            }
        }
        free(list->docs);
        list->docs = nullptr;
        list->count = 0;
    }
}

void zvec_write_results_free(zvec_write_results_t* results) {
    if (results) {
        for (size_t i = 0; i < results->count; i++) {
            zvec_status_free(&results->statuses[i]);
        }
        free(results->statuses);
        results->statuses = nullptr;
        results->count = 0;
    }
}

void zvec_doc_map_free(zvec_doc_map_t* map) {
    if (map) {
        for (size_t i = 0; i < map->count; i++) {
            free(map->keys[i]);
            if (map->docs[i]) {
                map->docs[i]->owned = false;
                zvec_doc_free(map->docs[i]);
            }
        }
        free(map->keys);
        free(map->docs);
        map->keys = nullptr;
        map->docs = nullptr;
        map->count = 0;
    }
}

}
