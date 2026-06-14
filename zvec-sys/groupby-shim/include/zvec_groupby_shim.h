#ifndef ZVEC_GROUPBY_SHIM_H
#define ZVEC_GROUPBY_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Minimal group-by shim for zvec-rust-bindings.
 *
 * The upstream zvec C API (v0.5.0) does not expose group-by query.
 * This shim wraps zvec::Collection::GroupByQuery directly so the Rust
 * bindings can preserve the group_by_query feature.
 *
 * All functions use the `zvecgb_*` prefix to avoid any future symbol
 * collision with upstream's `zvec_*` C API surface.
 *
 * The `collection_handle` parameter accepted by zvecgb_collection_group_by_query
 * is the same pointer returned by upstream's zvec_collection_create_and_open
 * (i.e. a std::shared_ptr<zvec::Collection>* cast to zvec_collection_t*).
 * The shim reinterprets it back to that type.
 */

typedef struct zvecgb_group_by_vector_query zvecgb_group_by_vector_query_t;
typedef struct zvecgb_group_results         zvecgb_group_results_t;

/* Create / destroy a group-by query. */
zvecgb_group_by_vector_query_t* zvecgb_group_by_vector_query_create(const char* field_name);
void zvecgb_group_by_vector_query_destroy(zvecgb_group_by_vector_query_t* q);

/* Builder setters. No-op if q is null. */
void zvecgb_group_by_vector_query_set_group_by_field(zvecgb_group_by_vector_query_t* q,
                                                     const char* field_name);
void zvecgb_group_by_vector_query_set_group_count(zvecgb_group_by_vector_query_t* q,
                                                  uint32_t count);
void zvecgb_group_by_vector_query_set_group_topk(zvecgb_group_by_vector_query_t* q,
                                                 uint32_t topk);
void zvecgb_group_by_vector_query_set_filter(zvecgb_group_by_vector_query_t* q,
                                             const char* filter);
void zvecgb_group_by_vector_query_set_output_fields(zvecgb_group_by_vector_query_t* q,
                                                    const char** fields, size_t count);

/* Sets the query vector as raw fp32 bytes (data[0..len*4]).
 * Returns 0 on success, non-zero zvec error code on invalid arguments. */
int zvecgb_group_by_vector_query_set_vector_fp32(zvecgb_group_by_vector_query_t* q,
                                                 const float* data, size_t len);

/* Returns 0 on success, non-zero zvec error code on failure.
 * On success, *out points to a freshly allocated zvecgb_group_results_t
 * that the caller must later pass to zvecgb_group_results_destroy.
 *
 * `collection_handle` is the zvec_collection_t* returned by upstream's
 * zvec_collection_create_and_open / zvec_collection_open. */
int zvecgb_collection_group_by_query(void* collection_handle,
                                     zvecgb_group_by_vector_query_t* q,
                                     zvecgb_group_results_t** out);

/* Result accessors. Return 0/NULL if results is null or index is out of range. */
size_t zvecgb_group_results_count(const zvecgb_group_results_t* r);
const char* zvecgb_group_results_group_by_value(const zvecgb_group_results_t* r, size_t i);

/* Per-group docs view: returns pointer to array of upstream zvec_doc_t*.
 * The docs are owned by `r` and freed when r is destroyed. */
void** zvecgb_group_results_docs_ptr(const zvecgb_group_results_t* r, size_t i);
size_t zvecgb_group_results_docs_count(const zvecgb_group_results_t* r, size_t i);

void zvecgb_group_results_destroy(zvecgb_group_results_t* r);

#ifdef __cplusplus
}
#endif

#endif /* ZVEC_GROUPBY_SHIM_H */
