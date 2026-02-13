#include "zvec_c_internal.h"
#include <cstring>

extern "C" {

zvec_vector_query_t* zvec_vector_query_new(const char* field_name) {
    auto* query = new zvec_vector_query_t;
    query->query.field_name_ = std::string(field_name);
    query->query.topk_ = 10;
    return query;
}

void zvec_vector_query_free(zvec_vector_query_t* query) {
    delete query;
}

void zvec_vector_query_set_topk(zvec_vector_query_t* query, int topk) {
    if (query) {
        query->query.topk_ = topk;
    }
}

void zvec_vector_query_set_filter(zvec_vector_query_t* query, const char* filter) {
    if (query && filter) {
        query->query.filter_ = std::string(filter);
    }
}

void zvec_vector_query_set_include_vector(zvec_vector_query_t* query, bool include) {
    if (query) {
        query->query.include_vector_ = include;
    }
}

void zvec_vector_query_set_include_doc_id(zvec_vector_query_t* query, bool include) {
    if (query) {
        query->query.include_doc_id_ = include;
    }
}

void zvec_vector_query_set_output_fields(zvec_vector_query_t* query, const char** fields, size_t count) {
    if (query && fields) {
        std::vector<std::string> out_fields;
        out_fields.reserve(count);
        for (size_t i = 0; i < count; i++) {
            out_fields.emplace_back(fields[i]);
        }
        query->query.output_fields_ = std::move(out_fields);
    }
}

void zvec_vector_query_set_query_params(zvec_vector_query_t* query, zvec_query_params_t* params) {
    if (query && params && params->ptr) {
        query->query.query_params_ = params->ptr;
    }
}

zvec_status_t zvec_vector_query_set_vector_fp32(zvec_vector_query_t* query, const float* data, size_t len) {
    if (!query || !data || len == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::string buf;
    buf.resize(len * sizeof(float));
    std::memcpy(&buf[0], data, len * sizeof(float));
    query->query.query_vector_ = std::move(buf);
    return zvec_wrapper::ok_status();
}

zvec_status_t zvec_vector_query_set_sparse_vector_fp32(zvec_vector_query_t* query,
    const uint32_t* indices, size_t indices_count, const float* values, size_t values_count) {
    if (!query || !indices || !values || indices_count != values_count) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::string idx_buf;
    idx_buf.resize(indices_count * sizeof(uint32_t));
    std::memcpy(&idx_buf[0], indices, indices_count * sizeof(uint32_t));
    query->query.query_sparse_indices_ = std::move(idx_buf);
    
    std::string val_buf;
    val_buf.resize(values_count * sizeof(float));
    std::memcpy(&val_buf[0], values, values_count * sizeof(float));
    query->query.query_sparse_values_ = std::move(val_buf);
    
    return zvec_wrapper::ok_status();
}

zvec_group_by_vector_query_t* zvec_group_by_vector_query_new(const char* field_name) {
    auto* query = new zvec_group_by_vector_query_t;
    query->query.field_name_ = std::string(field_name);
    return query;
}

void zvec_group_by_vector_query_free(zvec_group_by_vector_query_t* query) {
    delete query;
}

void zvec_group_by_vector_query_set_group_by_field(zvec_group_by_vector_query_t* query, const char* field_name) {
    if (query && field_name) {
        query->query.group_by_field_name_ = std::string(field_name);
    }
}

void zvec_group_by_vector_query_set_group_count(zvec_group_by_vector_query_t* query, uint32_t count) {
    if (query) {
        query->query.group_count_ = count;
    }
}

void zvec_group_by_vector_query_set_group_topk(zvec_group_by_vector_query_t* query, uint32_t topk) {
    if (query) {
        query->query.group_topk_ = topk;
    }
}

void zvec_group_by_vector_query_set_filter(zvec_group_by_vector_query_t* query, const char* filter) {
    if (query && filter) {
        query->query.filter_ = std::string(filter);
    }
}

void zvec_group_by_vector_query_set_output_fields(zvec_group_by_vector_query_t* query, 
    const char** fields, size_t count) {
    if (query && fields) {
        std::vector<std::string> out_fields;
        out_fields.reserve(count);
        for (size_t i = 0; i < count; i++) {
            out_fields.emplace_back(fields[i]);
        }
        query->query.output_fields_ = std::move(out_fields);
    }
}

zvec_status_t zvec_group_by_vector_query_set_vector_fp32(zvec_group_by_vector_query_t* query, 
    const float* data, size_t len) {
    if (!query || !data || len == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    std::string buf;
    buf.resize(len * sizeof(float));
    std::memcpy(&buf[0], data, len * sizeof(float));
    query->query.query_vector_ = std::move(buf);
    return zvec_wrapper::ok_status();
}

void zvec_group_results_free(zvec_group_results_t* results) {
    if (results) {
        for (size_t i = 0; i < results->count; i++) {
            if (results->groups[i].group_by_value) {
                free(results->groups[i].group_by_value);
            }
            zvec_doc_list_free(&results->groups[i].docs);
        }
        free(results->groups);
        results->groups = nullptr;
        results->count = 0;
    }
}

}
