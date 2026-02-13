#include "zvec_c_internal.h"

extern "C" {

zvec_index_params_t* zvec_index_params_new_hnsw(
    int m, int ef_construction, zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type) {
    auto* params = new zvec_index_params_t;
    auto metric = static_cast<zvec::MetricType>(static_cast<uint32_t>(metric_type));
    auto quantize = static_cast<zvec::QuantizeType>(static_cast<uint32_t>(quantize_type));
    params->ptr = std::make_shared<zvec::HnswIndexParams>(metric, m, ef_construction, quantize);
    return params;
}

zvec_index_params_t* zvec_index_params_new_ivf(
    int n_list, int n_iters, bool use_soar, zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type) {
    auto* params = new zvec_index_params_t;
    auto metric = static_cast<zvec::MetricType>(static_cast<uint32_t>(metric_type));
    auto quantize = static_cast<zvec::QuantizeType>(static_cast<uint32_t>(quantize_type));
    params->ptr = std::make_shared<zvec::IVFIndexParams>(metric, n_list, n_iters, use_soar, quantize);
    return params;
}

zvec_index_params_t* zvec_index_params_new_flat(
    zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type) {
    auto* params = new zvec_index_params_t;
    auto metric = static_cast<zvec::MetricType>(static_cast<uint32_t>(metric_type));
    auto quantize = static_cast<zvec::QuantizeType>(static_cast<uint32_t>(quantize_type));
    params->ptr = std::make_shared<zvec::FlatIndexParams>(metric, quantize);
    return params;
}

zvec_index_params_t* zvec_index_params_new_invert(bool enable_range_optimization) {
    auto* params = new zvec_index_params_t;
    params->ptr = std::make_shared<zvec::InvertIndexParams>(enable_range_optimization);
    return params;
}

void zvec_index_params_free(zvec_index_params_t* params) {
    delete params;
}

zvec_index_type_t zvec_index_params_type(const zvec_index_params_t* params) {
    if (params && params->ptr) {
        return static_cast<zvec_index_type_t>(static_cast<uint32_t>(params->ptr->type()));
    }
    return ZVEC_INDEX_TYPE_UNDEFINED;
}

zvec_query_params_t* zvec_query_params_new_hnsw(int ef_search) {
    auto* params = new zvec_query_params_t;
    params->ptr = std::make_shared<zvec::HnswQueryParams>(ef_search);
    return params;
}

zvec_query_params_t* zvec_query_params_new_ivf(int nprobe) {
    auto* params = new zvec_query_params_t;
    params->ptr = std::make_shared<zvec::IVFQueryParams>(nprobe);
    return params;
}

void zvec_query_params_free(zvec_query_params_t* params) {
    delete params;
}

}
