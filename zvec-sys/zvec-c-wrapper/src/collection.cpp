#include "zvec_c_internal.h"
#include <cstring>

extern "C" {

zvec_collection_t* zvec_collection_create_and_open(
    const char* path,
    zvec_collection_schema_t* schema,
    zvec_collection_options_t* options,
    zvec_status_t* out_status) {
    
    if (!path || !schema || !schema->ptr) {
        if (out_status) {
            out_status->code = ZVEC_STATUS_INVALID_ARGUMENT;
            out_status->message = strdup("Invalid path or schema");
        }
        return nullptr;
    }
    
    zvec::CollectionOptions opts;
    if (options) {
        opts = options->opts;
    }
    
    auto result = zvec::Collection::CreateAndOpen(std::string(path), *schema->ptr, opts);
    if (result.has_value()) {
        auto* collection = new zvec_collection_t;
        collection->ptr = result.value();
        if (out_status) {
            *out_status = zvec_wrapper::ok_status();
        }
        return collection;
    }
    
    if (out_status) {
        *out_status = zvec_wrapper::to_c_status(result.error());
    }
    return nullptr;
}

zvec_collection_t* zvec_collection_open(
    const char* path,
    zvec_collection_options_t* options,
    zvec_status_t* out_status) {
    
    if (!path) {
        if (out_status) {
            out_status->code = ZVEC_STATUS_INVALID_ARGUMENT;
            out_status->message = strdup("Invalid path");
        }
        return nullptr;
    }
    
    zvec::CollectionOptions opts;
    if (options) {
        opts = options->opts;
    }
    
    auto result = zvec::Collection::Open(std::string(path), opts);
    if (result.has_value()) {
        auto* collection = new zvec_collection_t;
        collection->ptr = result.value();
        if (out_status) {
            *out_status = zvec_wrapper::ok_status();
        }
        return collection;
    }
    
    if (out_status) {
        *out_status = zvec_wrapper::to_c_status(result.error());
    }
    return nullptr;
}

void zvec_collection_destroy(zvec_collection_t* collection) {
    delete collection;
}

zvec_status_t zvec_collection_path(const zvec_collection_t* collection, const char** out_path) {
    if (!collection || !collection->ptr || !out_path) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->Path();
    if (result.has_value()) {
        *out_path = strdup(result.value().c_str());
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_schema(const zvec_collection_t* collection, 
    zvec_collection_schema_t** out_schema) {
    if (!collection || !collection->ptr || !out_schema) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->Schema();
    if (result.has_value()) {
        auto* schema = new zvec_collection_schema_t;
        schema->ptr = std::make_shared<zvec::CollectionSchema>(result.value());
        schema->owned = true;
        *out_schema = schema;
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_options(const zvec_collection_t* collection,
    zvec_collection_options_t** out_options) {
    if (!collection || !collection->ptr || !out_options) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->Options();
    if (result.has_value()) {
        auto* opts = new zvec_collection_options_t;
        opts->opts = result.value();
        *out_options = opts;
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_stats(const zvec_collection_t* collection,
    zvec_collection_stats_t** out_stats) {
    if (!collection || !collection->ptr || !out_stats) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->Stats();
    if (result.has_value()) {
        auto* stats = new zvec_collection_stats_t;
        const auto& s = result.value();
        stats->doc_count = s.doc_count;
        stats->memory_usage = 0; // Not available in zvec::CollectionStats
        stats->json_details = nullptr;
        *out_stats = stats;
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

void zvec_collection_stats_free(zvec_collection_stats_t* stats) {
    if (stats) {
        if (stats->json_details) {
            free(stats->json_details);
        }
        delete stats;
    }
}

zvec_status_t zvec_collection_create_index(
    zvec_collection_t* collection,
    const char* column_name,
    zvec_index_params_t* index_params,
    zvec_create_index_options_t* options) {
    
    if (!collection || !collection->ptr || !column_name || !index_params || !index_params->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    zvec::CreateIndexOptions opts;
    if (options) {
        opts = options->opts;
    }
    
    auto status = collection->ptr->CreateIndex(std::string(column_name), index_params->ptr, opts);
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_drop_index(
    zvec_collection_t* collection,
    const char* column_name) {
    
    if (!collection || !collection->ptr || !column_name) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto status = collection->ptr->DropIndex(std::string(column_name));
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_optimize(
    zvec_collection_t* collection,
    zvec_optimize_options_t* options) {
    
    if (!collection || !collection->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid collection");
        return s;
    }
    
    zvec::OptimizeOptions opts;
    if (options) {
        opts = options->opts;
    }
    
    auto status = collection->ptr->Optimize(opts);
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_add_column(
    zvec_collection_t* collection,
    zvec_field_schema_t* column_schema,
    const char* expression) {
    
    if (!collection || !collection->ptr || !column_schema || !column_schema->ptr || !expression) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    zvec::AddColumnOptions opts;
    auto status = collection->ptr->AddColumn(
        column_schema->ptr,
        std::string(expression),
        opts
    );
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_drop_column(
    zvec_collection_t* collection,
    const char* column_name) {
    
    if (!collection || !collection->ptr || !column_name) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto status = collection->ptr->DropColumn(std::string(column_name));
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_alter_column(
    zvec_collection_t* collection,
    const char* column_name,
    const char* rename,
    zvec_field_schema_t* new_column_schema) {
    
    if (!collection || !collection->ptr || !column_name) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    zvec::AlterColumnOptions opts;
    zvec::FieldSchema::Ptr new_schema;
    if (new_column_schema && new_column_schema->ptr) {
        new_schema = new_column_schema->ptr;
    }
    
    auto status = collection->ptr->AlterColumn(
        std::string(column_name), 
        rename ? std::string(rename) : std::string(),
        new_schema, opts);
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_insert(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results) {
    
    if (!collection || !collection->ptr || !docs || count == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    std::vector<zvec::Doc> cpp_docs;
    cpp_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            cpp_docs.push_back(*docs[i]->ptr);
        }
    }
    
    auto result = collection->ptr->Insert(cpp_docs);
    if (result.has_value() && out_results) {
        const auto& write_results = result.value();
        out_results->count = write_results.size();
        out_results->statuses = (zvec_status_t*)malloc(sizeof(zvec_status_t) * write_results.size());
        for (size_t i = 0; i < write_results.size(); i++) {
            out_results->statuses[i] = zvec_wrapper::to_c_status(write_results[i]);
        }
    }
    
    return result.has_value() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_upsert(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results) {
    
    if (!collection || !collection->ptr || !docs || count == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    std::vector<zvec::Doc> cpp_docs;
    cpp_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            cpp_docs.push_back(*docs[i]->ptr);
        }
    }
    
    auto result = collection->ptr->Upsert(cpp_docs);
    if (result.has_value() && out_results) {
        const auto& write_results = result.value();
        out_results->count = write_results.size();
        out_results->statuses = (zvec_status_t*)malloc(sizeof(zvec_status_t) * write_results.size());
        for (size_t i = 0; i < write_results.size(); i++) {
            out_results->statuses[i] = zvec_wrapper::to_c_status(write_results[i]);
        }
    }
    
    return result.has_value() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_update(
    zvec_collection_t* collection,
    zvec_doc_t** docs,
    size_t count,
    zvec_write_results_t* out_results) {
    
    if (!collection || !collection->ptr || !docs || count == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    std::vector<zvec::Doc> cpp_docs;
    cpp_docs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        if (docs[i] && docs[i]->ptr) {
            cpp_docs.push_back(*docs[i]->ptr);
        }
    }
    
    auto result = collection->ptr->Update(cpp_docs);
    if (result.has_value() && out_results) {
        const auto& write_results = result.value();
        out_results->count = write_results.size();
        out_results->statuses = (zvec_status_t*)malloc(sizeof(zvec_status_t) * write_results.size());
        for (size_t i = 0; i < write_results.size(); i++) {
            out_results->statuses[i] = zvec_wrapper::to_c_status(write_results[i]);
        }
    }
    
    return result.has_value() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_delete(
    zvec_collection_t* collection,
    const char** pks,
    size_t count,
    zvec_write_results_t* out_results) {
    
    if (!collection || !collection->ptr || !pks || count == 0) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    std::vector<std::string> cpp_pks;
    cpp_pks.reserve(count);
    for (size_t i = 0; i < count; i++) {
        cpp_pks.emplace_back(pks[i]);
    }
    
    auto result = collection->ptr->Delete(cpp_pks);
    if (result.has_value() && out_results) {
        const auto& write_results = result.value();
        out_results->count = write_results.size();
        out_results->statuses = (zvec_status_t*)malloc(sizeof(zvec_status_t) * write_results.size());
        for (size_t i = 0; i < write_results.size(); i++) {
            out_results->statuses[i] = zvec_wrapper::to_c_status(write_results[i]);
        }
    }
    
    return result.has_value() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_delete_by_filter(
    zvec_collection_t* collection,
    const char* filter) {
    
    if (!collection || !collection->ptr || !filter) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto status = collection->ptr->DeleteByFilter(std::string(filter));
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_query(
    const zvec_collection_t* collection,
    zvec_vector_query_t* query,
    zvec_doc_list_t* out_results) {
    
    if (!collection || !collection->ptr || !query || !out_results) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->Query(query->query);
    if (result.has_value()) {
        const auto& docs = result.value();
        out_results->count = docs.size();
        out_results->docs = (zvec_doc_t**)malloc(sizeof(zvec_doc_t*) * docs.size());
        for (size_t i = 0; i < docs.size(); i++) {
            auto* doc = new zvec_doc_t;
            doc->ptr = docs[i];
            doc->owned = false;
            out_results->docs[i] = doc;
        }
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_group_by_query(
    const zvec_collection_t* collection,
    zvec_group_by_vector_query_t* query,
    zvec_group_results_t* out_results) {
    
    if (!collection || !collection->ptr || !query || !out_results) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    auto result = collection->ptr->GroupByQuery(query->query);
    if (result.has_value()) {
        const auto& groups = result.value();
        out_results->count = groups.size();
        out_results->groups = (zvec_group_result_t*)malloc(sizeof(zvec_group_result_t) * groups.size());
        for (size_t i = 0; i < groups.size(); i++) {
            out_results->groups[i].group_by_value = strdup(groups[i].group_by_value_.c_str());
            const auto& docs = groups[i].docs_;
            out_results->groups[i].docs.count = docs.size();
            out_results->groups[i].docs.docs = (zvec_doc_t**)malloc(sizeof(zvec_doc_t*) * docs.size());
            for (size_t j = 0; j < docs.size(); j++) {
                auto* doc = new zvec_doc_t;
                doc->ptr = std::make_shared<zvec::Doc>(docs[j]);
                doc->owned = true;
                out_results->groups[i].docs.docs[j] = doc;
            }
        }
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_fetch(
    const zvec_collection_t* collection,
    const char** pks,
    size_t count,
    zvec_doc_map_t* out_results) {
    
    if (!collection || !collection->ptr || !pks || count == 0 || !out_results) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    
    std::vector<std::string> cpp_pks;
    cpp_pks.reserve(count);
    for (size_t i = 0; i < count; i++) {
        cpp_pks.emplace_back(pks[i]);
    }
    
    auto result = collection->ptr->Fetch(cpp_pks);
    if (result.has_value()) {
        const auto& doc_map = result.value();
        out_results->count = doc_map.size();
        out_results->keys = (char**)malloc(sizeof(char*) * doc_map.size());
        out_results->docs = (zvec_doc_t**)malloc(sizeof(zvec_doc_t*) * doc_map.size());
        size_t idx = 0;
        for (const auto& [pk, doc_ptr] : doc_map) {
            out_results->keys[idx] = strdup(pk.c_str());
            auto* doc = new zvec_doc_t;
            doc->ptr = doc_ptr;
            doc->owned = false;
            out_results->docs[idx] = doc;
            idx++;
        }
        return zvec_wrapper::ok_status();
    }
    return zvec_wrapper::to_c_status(result.error());
}

zvec_status_t zvec_collection_flush(zvec_collection_t* collection) {
    if (!collection || !collection->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid collection");
        return s;
    }
    
    auto status = collection->ptr->Flush();
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_destroy_storage(zvec_collection_t* collection) {
    if (!collection || !collection->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid collection");
        return s;
    }
    
    auto status = collection->ptr->Destroy();
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

void zvec_set_log_level(int level) {
    (void)level;
}

void zvec_set_thread_pool_size(size_t size) {
    (void)size;
}

}
