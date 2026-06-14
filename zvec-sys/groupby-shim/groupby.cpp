// Minimal group-by shim for zvec-rust-bindings.
//
// The upstream zvec C API (v0.5.0) does not expose group-by query.
// This shim wraps zvec::Collection::GroupByQuery directly so the Rust
// bindings can preserve the group_by_query feature.
//
// `collection_handle` is the zvec_collection_t* produced by upstream's
// zvec_collection_create_and_open, which the upstream c_api.cc stores as
// a heap-allocated `std::shared_ptr<zvec::Collection>*`.

#include "zvec_groupby_shim.h"

#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/query.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

// Upstream zvec_collection_t layout (per src/binding/c/c_api.cc):
//   new std::shared_ptr<zvec::Collection>(...)
// So a zvec_collection_t* == std::shared_ptr<zvec::Collection>*.
std::shared_ptr<zvec::Collection> deref_collection(void* handle) {
  if (!handle) return nullptr;
  auto* sp = reinterpret_cast<std::shared_ptr<zvec::Collection>*>(handle);
  return *sp;
}

}  // namespace

extern "C" {

struct zvecgb_group_by_vector_query {
  zvec::GroupByVectorQuery q;
};

struct zvecgb_group_results {
  // Each GroupResult owns its docs (vector<Doc>). We materialize shared_ptrs
  // to those Docs so the caller can hold stable pointers without copying.
  std::vector<std::string> group_by_values;            // per-group owning storage
  std::vector<std::vector<zvec::Doc::Ptr>> docs_per_group;
  // Flat doc handle array exposed via zvecgb_group_results_docs_ptr.
  // For group i, the slice [offsets[i], offsets[i] + sizes[i]) is valid.
  std::vector<std::vector<void*>> docs_ptr_per_group;
};

zvecgb_group_by_vector_query_t* zvecgb_group_by_vector_query_create(const char* field_name) {
  if (!field_name) return nullptr;
  auto* q = new (std::nothrow) zvecgb_group_by_vector_query_t();
  if (!q) return nullptr;
  q->q.target_.field_name_ = std::string(field_name);
  return q;
}

void zvecgb_group_by_vector_query_destroy(zvecgb_group_by_vector_query_t* q) {
  delete q;
}

void zvecgb_group_by_vector_query_set_group_by_field(zvecgb_group_by_vector_query_t* q,
                                                     const char* field_name) {
  if (q && field_name) q->q.group_by_field_name_ = std::string(field_name);
}

void zvecgb_group_by_vector_query_set_group_count(zvecgb_group_by_vector_query_t* q,
                                                  uint32_t count) {
  if (q) q->q.group_count_ = count;
}

void zvecgb_group_by_vector_query_set_group_topk(zvecgb_group_by_vector_query_t* q,
                                                 uint32_t topk) {
  if (q) q->q.group_topk_ = topk;
}

void zvecgb_group_by_vector_query_set_filter(zvecgb_group_by_vector_query_t* q,
                                             const char* filter) {
  if (q && filter) q->q.filter_ = std::string(filter);
}

void zvecgb_group_by_vector_query_set_output_fields(zvecgb_group_by_vector_query_t* q,
                                                    const char** fields, size_t count) {
  if (!q || !fields) return;
  std::vector<std::string> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (fields[i]) out.emplace_back(fields[i]);
  }
  q->q.output_fields_ = std::move(out);
}

int zvecgb_group_by_vector_query_set_vector_fp32(zvecgb_group_by_vector_query_t* q,
                                                 const float* data, size_t len) {
  if (!q || !data || len == 0) return 3;  // ZVEC_ERROR_INVALID_ARGUMENT
  std::string buf;
  buf.resize(len * sizeof(float));
  std::memcpy(&buf[0], data, len * sizeof(float));
  q->q.target_.set_vector(std::move(buf));
  return 0;
}

int zvecgb_collection_group_by_query(void* collection_handle,
                                     zvecgb_group_by_vector_query_t* q,
                                     zvecgb_group_results_t** out) {
  if (!collection_handle || !q || !out) return 4;  // ZVEC_ERROR_INVALID_ARGUMENT=3;
                                                   // use a generic invalid-arg code.

  auto coll = deref_collection(collection_handle);
  if (!coll) return 4;

  auto result = coll->GroupByQuery(q->q);
  if (!result.has_value()) {
    // Map zvec::Status code to upstream zvec_error_code_t (same numeric values
    // for the common cases: NotFound=1, AlreadyExists=2, InvalidArgument=3, ...).
    return static_cast<int>(result.error().code());
  }

  const auto& groups = result.value();
  auto* r = new (std::nothrow) zvecgb_group_results_t();
  if (!r) return 8;  // ZVEC_ERROR_INTERNAL_ERROR=8

  r->group_by_values.reserve(groups.size());
  r->docs_per_group.reserve(groups.size());
  r->docs_ptr_per_group.reserve(groups.size());

  for (const auto& g : groups) {
    r->group_by_values.push_back(g.group_by_value_);

    std::vector<zvec::Doc::Ptr> doc_ptrs;
    std::vector<void*> doc_handles;
    doc_ptrs.reserve(g.docs_.size());
    doc_handles.reserve(g.docs_.size());
    for (const auto& d : g.docs_) {
      // Wrap each returned Doc in a shared_ptr that the Rust side will free
      // via zvec_doc_destroy (upstream's API expects zvec_doc_t* which is
      // a zvec::Doc*; upstream's zvec_doc_destroy calls `delete` on it).
      // To match upstream ownership, we allocate a *copy* on the heap that
      // the caller owns.
      auto* heap_doc = new (std::nothrow) zvec::Doc(d);
      if (!heap_doc) {
        // Allocation failure: bail; caller will see a non-zero code.
        delete r;
        return 8;
      }
      doc_handles.push_back(reinterpret_cast<void*>(heap_doc));
      doc_ptrs.emplace_back(heap_doc);  // kept for our bookkeeping; not used downstream
    }
    r->docs_per_group.push_back(std::move(doc_ptrs));
    r->docs_ptr_per_group.push_back(std::move(doc_handles));
  }

  *out = r;
  return 0;  // ZVEC_OK
}

size_t zvecgb_group_results_count(const zvecgb_group_results_t* r) {
  return r ? r->group_by_values.size() : 0;
}

const char* zvecgb_group_results_group_by_value(const zvecgb_group_results_t* r, size_t i) {
  if (!r || i >= r->group_by_values.size()) return nullptr;
  return r->group_by_values[i].c_str();
}

void** zvecgb_group_results_docs_ptr(const zvecgb_group_results_t* r, size_t i) {
  if (!r || i >= r->docs_ptr_per_group.size()) return nullptr;
  return const_cast<void**>(r->docs_ptr_per_group[i].data());
}

size_t zvecgb_group_results_docs_count(const zvecgb_group_results_t* r, size_t i) {
  if (!r || i >= r->docs_per_group.size()) return 0;
  return r->docs_per_group[i].size();
}

void zvecgb_group_results_destroy(zvecgb_group_results_t* r) {
  if (!r) return;
  // Each Doc was heap-allocated in zvecgb_collection_group_by_query. The Rust
  // side does NOT own these individually (it borrows them via DocRef); so we
  // must delete them here along with the result container.
  for (auto& group : r->docs_per_group) {
    for (auto& ptr : group) {
      ptr.reset();  // releases the shared_ptr; underlying Doc freed when refcount=0
    }
  }
  delete r;
}

}  // extern "C"
