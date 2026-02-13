#pragma once

#include "zvec_c.h"
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include <zvec/db/index_params.h>
#include <zvec/db/query_params.h>
#include <zvec/db/options.h>
#include <memory>
#include <string>
#include <cstdlib>

namespace zvec_wrapper {

inline zvec_status_t ok_status() {
    zvec_status_t s;
    s.code = ZVEC_STATUS_OK;
    s.message = nullptr;
    return s;
}

inline zvec_status_t to_c_status(const zvec::Status& status) {
    zvec_status_t s;
    s.code = static_cast<zvec_status_code_t>(static_cast<int>(status.code()));
    s.message = status.message().empty() ? nullptr : strdup(status.message().c_str());
    return s;
}

inline zvec::DataType to_cpp_data_type(zvec_data_type_t t) {
    return static_cast<zvec::DataType>(static_cast<uint32_t>(t));
}

inline zvec_data_type_t to_c_data_type(zvec::DataType t) {
    return static_cast<zvec_data_type_t>(static_cast<uint32_t>(t));
}

}

extern "C" {

struct zvec_collection {
    zvec::Collection::Ptr ptr;
};

struct zvec_collection_schema {
    zvec::CollectionSchema::Ptr ptr;
    bool owned;
};

struct zvec_field_schema {
    zvec::FieldSchema::Ptr ptr;
    bool owned;
};

struct zvec_doc {
    zvec::Doc::Ptr ptr;
    bool owned;
};

struct zvec_vector_query {
    zvec::VectorQuery query;
};

struct zvec_group_by_vector_query {
    zvec::GroupByVectorQuery query;
};

struct zvec_index_params {
    zvec::IndexParams::Ptr ptr;
};

struct zvec_query_params {
    zvec::QueryParams::Ptr ptr;
};

struct zvec_collection_options {
    zvec::CollectionOptions opts;
};

struct zvec_create_index_options {
    zvec::CreateIndexOptions opts;
};

struct zvec_optimize_options {
    zvec::OptimizeOptions opts;
};

}
