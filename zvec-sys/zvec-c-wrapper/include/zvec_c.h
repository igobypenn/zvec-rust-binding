/**
 * zvec C API - C wrapper for zvec C++ vector database library
 * 
 * This provides a C-compatible API for the zvec database layer,
 * enabling bindings to other languages (Rust, etc.)
 */

#ifndef ZVEC_C_H
#define ZVEC_C_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Definitions (Opaque Handles)
 * ============================================================================ */

typedef struct zvec_collection zvec_collection_t;
typedef struct zvec_collection_schema zvec_collection_schema_t;
typedef struct zvec_field_schema zvec_field_schema_t;
typedef struct zvec_doc zvec_doc_t;
typedef struct zvec_vector_query zvec_vector_query_t;
typedef struct zvec_group_by_vector_query zvec_group_by_vector_query_t;
typedef struct zvec_index_params zvec_index_params_t;
typedef struct zvec_query_params zvec_query_params_t;
typedef struct zvec_collection_options zvec_collection_options_t;
typedef struct zvec_create_index_options zvec_create_index_options_t;
typedef struct zvec_optimize_options zvec_optimize_options_t;
typedef struct zvec_collection_stats zvec_collection_stats_t;

/* ============================================================================
 * Enums
 * ============================================================================ */

typedef enum zvec_status_code {
    ZVEC_STATUS_OK = 0,
    ZVEC_STATUS_NOT_FOUND = 1,
    ZVEC_STATUS_ALREADY_EXISTS = 2,
    ZVEC_STATUS_INVALID_ARGUMENT = 3,
    ZVEC_STATUS_NOT_SUPPORTED = 4,
    ZVEC_STATUS_INTERNAL_ERROR = 5,
    ZVEC_STATUS_PERMISSION_DENIED = 6,
    ZVEC_STATUS_FAILED_PRECONDITION = 7,
    ZVEC_STATUS_UNKNOWN = 8
} zvec_status_code_t;

typedef enum zvec_data_type {
    ZVEC_DATA_TYPE_UNDEFINED = 0,
    ZVEC_DATA_TYPE_BINARY = 1,
    ZVEC_DATA_TYPE_STRING = 2,
    ZVEC_DATA_TYPE_BOOL = 3,
    ZVEC_DATA_TYPE_INT32 = 4,
    ZVEC_DATA_TYPE_INT64 = 5,
    ZVEC_DATA_TYPE_UINT32 = 6,
    ZVEC_DATA_TYPE_UINT64 = 7,
    ZVEC_DATA_TYPE_FLOAT = 8,
    ZVEC_DATA_TYPE_DOUBLE = 9,
    ZVEC_DATA_TYPE_VECTOR_BINARY32 = 20,
    ZVEC_DATA_TYPE_VECTOR_BINARY64 = 21,
    ZVEC_DATA_TYPE_VECTOR_FP16 = 22,
    ZVEC_DATA_TYPE_VECTOR_FP32 = 23,
    ZVEC_DATA_TYPE_VECTOR_FP64 = 24,
    ZVEC_DATA_TYPE_VECTOR_INT4 = 25,
    ZVEC_DATA_TYPE_VECTOR_INT8 = 26,
    ZVEC_DATA_TYPE_VECTOR_INT16 = 27,
    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP16 = 30,
    ZVEC_DATA_TYPE_SPARSE_VECTOR_FP32 = 31,
    ZVEC_DATA_TYPE_ARRAY_BINARY = 40,
    ZVEC_DATA_TYPE_ARRAY_STRING = 41,
    ZVEC_DATA_TYPE_ARRAY_BOOL = 42,
    ZVEC_DATA_TYPE_ARRAY_INT32 = 43,
    ZVEC_DATA_TYPE_ARRAY_INT64 = 44,
    ZVEC_DATA_TYPE_ARRAY_UINT32 = 45,
    ZVEC_DATA_TYPE_ARRAY_UINT64 = 46,
    ZVEC_DATA_TYPE_ARRAY_FLOAT = 47,
    ZVEC_DATA_TYPE_ARRAY_DOUBLE = 48
} zvec_data_type_t;

typedef enum zvec_index_type {
    ZVEC_INDEX_TYPE_UNDEFINED = 0,
    ZVEC_INDEX_TYPE_HNSW = 1,
    ZVEC_INDEX_TYPE_IVF = 3,
    ZVEC_INDEX_TYPE_FLAT = 4,
    ZVEC_INDEX_TYPE_INVERT = 10
} zvec_index_type_t;

typedef enum zvec_metric_type {
    ZVEC_METRIC_TYPE_UNDEFINED = 0,
    ZVEC_METRIC_TYPE_L2 = 1,
    ZVEC_METRIC_TYPE_IP = 2,
    ZVEC_METRIC_TYPE_COSINE = 3,
    ZVEC_METRIC_TYPE_MIPS_L2 = 4
} zvec_metric_type_t;

typedef enum zvec_quantize_type {
    ZVEC_QUANTIZE_TYPE_UNDEFINED = 0,
    ZVEC_QUANTIZE_TYPE_FP16 = 1,
    ZVEC_QUANTIZE_TYPE_INT8 = 2,
    ZVEC_QUANTIZE_TYPE_INT4 = 3
} zvec_quantize_type_t;

typedef enum zvec_operator {
    ZVEC_OPERATOR_INSERT = 0,
    ZVEC_OPERATOR_UPSERT = 1,
    ZVEC_OPERATOR_UPDATE = 2,
    ZVEC_OPERATOR_DELETE = 3
} zvec_operator_t;

/* ============================================================================
 * Status / Error Handling
 * ============================================================================ */

typedef struct zvec_status {
    zvec_status_code_t code;
    const char* message;
} zvec_status_t;

zvec_status_t zvec_status_ok(void);
bool zvec_status_is_ok(const zvec_status_t* status);
void zvec_status_free(zvec_status_t* status);

/* ============================================================================
 * String Array (for returning lists of strings)
 * ============================================================================ */

typedef struct zvec_string_array {
    char** strings;
    size_t count;
} zvec_string_array_t;

void zvec_string_array_free(zvec_string_array_t* arr);

/* ============================================================================
 * Initialization / Debug
 * ============================================================================ */

bool zvec_init(void);
int zvec_list_registered_metrics(const char*** out_metrics);
int zvec_list_registered_builders(const char*** out_builders);
int zvec_list_registered_searchers(const char*** out_searchers);
int zvec_list_registered_streamers(const char*** out_streamers);

/* ============================================================================
 * Collection Options
 * ============================================================================ */

zvec_collection_options_t* zvec_collection_options_new(void);
void zvec_collection_options_free(zvec_collection_options_t* options);

void zvec_collection_options_set_read_only(zvec_collection_options_t* options, bool read_only);
void zvec_collection_options_set_enable_mmap(zvec_collection_options_t* options, bool enable_mmap);
void zvec_collection_options_set_max_buffer_size(zvec_collection_options_t* options, uint64_t max_buffer_size);

/* ============================================================================
 * Field Schema
 * ============================================================================ */

zvec_field_schema_t* zvec_field_schema_new(const char* name, zvec_data_type_t data_type);
zvec_field_schema_t* zvec_field_schema_new_with_dimension(
    const char* name, zvec_data_type_t data_type, uint32_t dimension);
void zvec_field_schema_free(zvec_field_schema_t* schema);

void zvec_field_schema_set_nullable(zvec_field_schema_t* schema, bool nullable);
void zvec_field_schema_set_dimension(zvec_field_schema_t* schema, uint32_t dimension);
void zvec_field_schema_set_index_params(zvec_field_schema_t* schema, zvec_index_params_t* params);

const char* zvec_field_schema_name(const zvec_field_schema_t* schema);
zvec_data_type_t zvec_field_schema_data_type(const zvec_field_schema_t* schema);
bool zvec_field_schema_nullable(const zvec_field_schema_t* schema);
uint32_t zvec_field_schema_dimension(const zvec_field_schema_t* schema);

/* ============================================================================
 * Collection Schema
 * ============================================================================ */

zvec_collection_schema_t* zvec_collection_schema_new(const char* name);
void zvec_collection_schema_free(zvec_collection_schema_t* schema);

zvec_status_t zvec_collection_schema_add_field(zvec_collection_schema_t* schema, zvec_field_schema_t* field);
zvec_status_t zvec_collection_schema_add_index(zvec_collection_schema_t* schema, 
    const char* column_name, zvec_index_params_t* params);

const char* zvec_collection_schema_name(const zvec_collection_schema_t* schema);
zvec_string_array_t zvec_collection_schema_field_names(const zvec_collection_schema_t* schema);
zvec_string_array_t zvec_collection_schema_vector_field_names(const zvec_collection_schema_t* schema);

/* ============================================================================
 * Index Parameters
 * ============================================================================ */

zvec_index_params_t* zvec_index_params_new_hnsw(
    int m, int ef_construction, zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type);
zvec_index_params_t* zvec_index_params_new_ivf(
    int n_list, int n_iters, bool use_soar, zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type);
zvec_index_params_t* zvec_index_params_new_flat(
    zvec_metric_type_t metric_type, zvec_quantize_type_t quantize_type);
zvec_index_params_t* zvec_index_params_new_invert(bool enable_range_optimization);
void zvec_index_params_free(zvec_index_params_t* params);

zvec_index_type_t zvec_index_params_type(const zvec_index_params_t* params);

/* ============================================================================
 * Query Parameters
 * ============================================================================ */

zvec_query_params_t* zvec_query_params_new_hnsw(int ef_search);
zvec_query_params_t* zvec_query_params_new_ivf(int nprobe);
void zvec_query_params_free(zvec_query_params_t* params);

/* ============================================================================
 * Create Index Options
 * ============================================================================ */

zvec_create_index_options_t* zvec_create_index_options_new(void);
void zvec_create_index_options_free(zvec_create_index_options_t* options);
void zvec_create_index_options_set_concurrency(zvec_create_index_options_t* options, int concurrency);

/* ============================================================================
 * Optimize Options
 * ============================================================================ */

zvec_optimize_options_t* zvec_optimize_options_new(void);
void zvec_optimize_options_free(zvec_optimize_options_t* options);
void zvec_optimize_options_set_concurrency(zvec_optimize_options_t* options, int concurrency);

/* ============================================================================
 * Document
 * ============================================================================ */

zvec_doc_t* zvec_doc_new(void);
void zvec_doc_free(zvec_doc_t* doc);

void zvec_doc_set_pk(zvec_doc_t* doc, const char* pk);
const char* zvec_doc_pk(const zvec_doc_t* doc);

void zvec_doc_set_score(zvec_doc_t* doc, float score);
float zvec_doc_score(const zvec_doc_t* doc);

void zvec_doc_set_doc_id(zvec_doc_t* doc, uint64_t doc_id);
uint64_t zvec_doc_doc_id(const zvec_doc_t* doc);

/* Scalar field setters */
zvec_status_t zvec_doc_set_bool(zvec_doc_t* doc, const char* field, bool value);
zvec_status_t zvec_doc_set_int32(zvec_doc_t* doc, const char* field, int32_t value);
zvec_status_t zvec_doc_set_int64(zvec_doc_t* doc, const char* field, int64_t value);
zvec_status_t zvec_doc_set_uint32(zvec_doc_t* doc, const char* field, uint32_t value);
zvec_status_t zvec_doc_set_uint64(zvec_doc_t* doc, const char* field, uint64_t value);
zvec_status_t zvec_doc_set_float(zvec_doc_t* doc, const char* field, float value);
zvec_status_t zvec_doc_set_double(zvec_doc_t* doc, const char* field, double value);
zvec_status_t zvec_doc_set_string(zvec_doc_t* doc, const char* field, const char* value);
void zvec_doc_set_null(zvec_doc_t* doc, const char* field);

/* Vector field setters */
zvec_status_t zvec_doc_set_vector_fp32(zvec_doc_t* doc, const char* field, const float* data, size_t len);
zvec_status_t zvec_doc_set_vector_fp64(zvec_doc_t* doc, const char* field, const double* data, size_t len);
zvec_status_t zvec_doc_set_vector_int8(zvec_doc_t* doc, const char* field, const int8_t* data, size_t len);
zvec_status_t zvec_doc_set_vector_int16(zvec_doc_t* doc, const char* field, const int16_t* data, size_t len);
zvec_status_t zvec_doc_set_vector_int32(zvec_doc_t* doc, const char* field, const int32_t* data, size_t len);
zvec_status_t zvec_doc_set_vector_int64(zvec_doc_t* doc, const char* field, const int64_t* data, size_t len);

/* Sparse vector setter */
zvec_status_t zvec_doc_set_sparse_vector_fp32(zvec_doc_t* doc, const char* field,
    const uint32_t* indices, size_t indices_count, const float* values, size_t values_count);

/* Array field setters */
zvec_status_t zvec_doc_set_array_int32(zvec_doc_t* doc, const char* field, const int32_t* data, size_t len);
zvec_status_t zvec_doc_set_array_int64(zvec_doc_t* doc, const char* field, const int64_t* data, size_t len);
zvec_status_t zvec_doc_set_array_float(zvec_doc_t* doc, const char* field, const float* data, size_t len);
zvec_status_t zvec_doc_set_array_double(zvec_doc_t* doc, const char* field, const double* data, size_t len);
zvec_status_t zvec_doc_set_array_string(zvec_doc_t* doc, const char* field, const char** data, size_t len);

/* Field getters - returns true if field exists and type matches */
bool zvec_doc_get_bool(const zvec_doc_t* doc, const char* field, bool* out_value);
bool zvec_doc_get_int32(const zvec_doc_t* doc, const char* field, int32_t* out_value);
bool zvec_doc_get_int64(const zvec_doc_t* doc, const char* field, int64_t* out_value);
bool zvec_doc_get_float(const zvec_doc_t* doc, const char* field, float* out_value);
bool zvec_doc_get_double(const zvec_doc_t* doc, const char* field, double* out_value);
bool zvec_doc_get_string(const zvec_doc_t* doc, const char* field, const char** out_value);

/* Vector getters - returns length, or 0 if not found */
size_t zvec_doc_get_vector_fp32(const zvec_doc_t* doc, const char* field, float* out_data, size_t max_len);

/* Field info */
bool zvec_doc_has(const zvec_doc_t* doc, const char* field);
bool zvec_doc_has_value(const zvec_doc_t* doc, const char* field);
bool zvec_doc_is_null(const zvec_doc_t* doc, const char* field);
zvec_string_array_t zvec_doc_field_names(const zvec_doc_t* doc);

/* ============================================================================
 * Doc List (for returning query results)
 * ============================================================================ */

typedef struct zvec_doc_list {
    zvec_doc_t** docs;
    size_t count;
} zvec_doc_list_t;

void zvec_doc_list_free(zvec_doc_list_t* list);

/* ============================================================================
 * Write Results (for insert/update/upsert/delete)
 * ============================================================================ */

typedef struct zvec_write_results {
    zvec_status_t* statuses;
    size_t count;
} zvec_write_results_t;

void zvec_write_results_free(zvec_write_results_t* results);

/* ============================================================================
 * Doc Map (for fetch results)
 * ============================================================================ */

typedef struct zvec_doc_map {
    char** keys;
    zvec_doc_t** docs;
    size_t count;
} zvec_doc_map_t;

void zvec_doc_map_free(zvec_doc_map_t* map);

/* ============================================================================
 * Vector Query
 * ============================================================================ */

zvec_vector_query_t* zvec_vector_query_new(const char* field_name);
void zvec_vector_query_free(zvec_vector_query_t* query);

void zvec_vector_query_set_topk(zvec_vector_query_t* query, int topk);
void zvec_vector_query_set_filter(zvec_vector_query_t* query, const char* filter);
void zvec_vector_query_set_include_vector(zvec_vector_query_t* query, bool include);
void zvec_vector_query_set_include_doc_id(zvec_vector_query_t* query, bool include);
void zvec_vector_query_set_output_fields(zvec_vector_query_t* query, const char** fields, size_t count);
void zvec_vector_query_set_query_params(zvec_vector_query_t* query, zvec_query_params_t* params);

/* Vector query input setters */
zvec_status_t zvec_vector_query_set_vector_fp32(zvec_vector_query_t* query, const float* data, size_t len);
zvec_status_t zvec_vector_query_set_sparse_vector_fp32(zvec_vector_query_t* query,
    const uint32_t* indices, size_t indices_count, const float* values, size_t values_count);

/* ============================================================================
 * Group By Vector Query
 * ============================================================================ */

zvec_group_by_vector_query_t* zvec_group_by_vector_query_new(const char* field_name);
void zvec_group_by_vector_query_free(zvec_group_by_vector_query_t* query);

void zvec_group_by_vector_query_set_group_by_field(zvec_group_by_vector_query_t* query, const char* field_name);
void zvec_group_by_vector_query_set_group_count(zvec_group_by_vector_query_t* query, uint32_t count);
void zvec_group_by_vector_query_set_group_topk(zvec_group_by_vector_query_t* query, uint32_t topk);
void zvec_group_by_vector_query_set_filter(zvec_group_by_vector_query_t* query, const char* filter);
void zvec_group_by_vector_query_set_output_fields(zvec_group_by_vector_query_t* query, 
    const char** fields, size_t count);

zvec_status_t zvec_group_by_vector_query_set_vector_fp32(zvec_group_by_vector_query_t* query, 
    const float* data, size_t len);

/* ============================================================================
 * Group Results
 * ============================================================================ */

typedef struct zvec_group_result {
    char* group_by_value;
    zvec_doc_list_t docs;
} zvec_group_result_t;

typedef struct zvec_group_results {
    zvec_group_result_t* groups;
    size_t count;
} zvec_group_results_t;

void zvec_group_results_free(zvec_group_results_t* results);

/* ============================================================================
 * Collection Stats
 * ============================================================================ */

typedef struct zvec_collection_stats {
    uint64_t doc_count;
    uint64_t memory_usage;
    char* json_details;
} zvec_collection_stats_t;

void zvec_collection_stats_free(zvec_collection_stats_t* stats);

/* ============================================================================
 * Collection - Lifecycle
 * ============================================================================ */

zvec_collection_t* zvec_collection_create_and_open(
    const char* path,
    zvec_collection_schema_t* schema,
    zvec_collection_options_t* options,
    zvec_status_t* out_status);

zvec_collection_t* zvec_collection_open(
    const char* path,
    zvec_collection_options_t* options,
    zvec_status_t* out_status);

void zvec_collection_destroy(zvec_collection_t* collection);

/* ============================================================================
 * Collection - Properties
 * ============================================================================ */

zvec_status_t zvec_collection_path(const zvec_collection_t* collection, const char** out_path);
zvec_status_t zvec_collection_schema(const zvec_collection_t* collection, 
    zvec_collection_schema_t** out_schema);
zvec_status_t zvec_collection_options(const zvec_collection_t* collection,
    zvec_collection_options_t** out_options);
zvec_status_t zvec_collection_stats(const zvec_collection_t* collection,
    zvec_collection_stats_t** out_stats);

/* ============================================================================
 * Collection - DDL Operations
 * ============================================================================ */

zvec_status_t zvec_collection_create_index(
    zvec_collection_t* collection,
    const char* column_name,
    zvec_index_params_t* index_params,
    zvec_create_index_options_t* options);

zvec_status_t zvec_collection_drop_index(
    zvec_collection_t* collection,
    const char* column_name);

zvec_status_t zvec_collection_optimize(
    zvec_collection_t* collection,
    zvec_optimize_options_t* options);

zvec_status_t zvec_collection_add_column(
    zvec_collection_t* collection,
    zvec_field_schema_t* column_schema,
    const char* expression);

zvec_status_t zvec_collection_drop_column(
    zvec_collection_t* collection,
    const char* column_name);

zvec_status_t zvec_collection_alter_column(
    zvec_collection_t* collection,
    const char* column_name,
    const char* rename,
    zvec_field_schema_t* new_column_schema);

/* ============================================================================
 * Collection - DML Operations
 * ============================================================================ */

zvec_status_t zvec_collection_insert(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results);

zvec_status_t zvec_collection_upsert(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results);

zvec_status_t zvec_collection_update(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results);

zvec_status_t zvec_collection_delete(
    zvec_collection_t* collection,
    const char** pks,
    size_t count,
    zvec_write_results_t* out_results);

zvec_status_t zvec_collection_delete_by_filter(
    zvec_collection_t* collection,
    const char* filter);

/* ============================================================================
 * Collection - DQL Operations
 * ============================================================================ */

zvec_status_t zvec_collection_query(
    const zvec_collection_t* collection,
    zvec_vector_query_t* query,
    zvec_doc_list_t* out_results);

zvec_status_t zvec_collection_group_by_query(
    const zvec_collection_t* collection,
    zvec_group_by_vector_query_t* query,
    zvec_group_results_t* out_results);

zvec_status_t zvec_collection_fetch(
    const zvec_collection_t* collection,
    const char** pks,
    size_t count,
    zvec_doc_map_t* out_results);

/* ============================================================================
 * Collection - Utility
 * ============================================================================ */

zvec_status_t zvec_collection_flush(zvec_collection_t* collection);
zvec_status_t zvec_collection_destroy_storage(zvec_collection_t* collection);

/* ============================================================================
 * Global Configuration
 * ============================================================================ */

void zvec_set_log_level(int level);
void zvec_set_thread_pool_size(size_t size);

#ifdef __cplusplus
}
#endif

#endif /* ZVEC_C_H */
