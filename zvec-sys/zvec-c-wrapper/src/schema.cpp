#include "zvec_c_internal.h"

extern "C" {

zvec_field_schema_t* zvec_field_schema_new(const char* name, zvec_data_type_t data_type) {
    auto* schema = new zvec_field_schema_t;
    schema->ptr = std::make_shared<zvec::FieldSchema>(std::string(name), zvec_wrapper::to_cpp_data_type(data_type));
    schema->owned = true;
    return schema;
}

zvec_field_schema_t* zvec_field_schema_new_with_dimension(
    const char* name, zvec_data_type_t data_type, uint32_t dimension) {
    auto* schema = new zvec_field_schema_t;
    schema->ptr = std::make_shared<zvec::FieldSchema>(
        std::string(name), zvec_wrapper::to_cpp_data_type(data_type), dimension, false);
    schema->owned = true;
    return schema;
}

void zvec_field_schema_free(zvec_field_schema_t* schema) {
    if (schema && schema->owned) {
        delete schema;
    }
}

void zvec_field_schema_set_nullable(zvec_field_schema_t* schema, bool nullable) {
    if (schema && schema->ptr) {
        schema->ptr->set_nullable(nullable);
    }
}

void zvec_field_schema_set_dimension(zvec_field_schema_t* schema, uint32_t dimension) {
    if (schema && schema->ptr) {
        schema->ptr->set_dimension(dimension);
    }
}

void zvec_field_schema_set_index_params(zvec_field_schema_t* schema, zvec_index_params_t* params) {
    if (schema && schema->ptr && params && params->ptr) {
        schema->ptr->set_index_params(params->ptr);
    }
}

const char* zvec_field_schema_name(const zvec_field_schema_t* schema) {
    if (schema && schema->ptr) {
        return schema->ptr->name().c_str();
    }
    return nullptr;
}

zvec_data_type_t zvec_field_schema_data_type(const zvec_field_schema_t* schema) {
    if (schema && schema->ptr) {
        return zvec_wrapper::to_c_data_type(schema->ptr->data_type());
    }
    return ZVEC_DATA_TYPE_UNDEFINED;
}

bool zvec_field_schema_nullable(const zvec_field_schema_t* schema) {
    if (schema && schema->ptr) {
        return schema->ptr->nullable();
    }
    return false;
}

uint32_t zvec_field_schema_dimension(const zvec_field_schema_t* schema) {
    if (schema && schema->ptr) {
        return schema->ptr->dimension();
    }
    return 0;
}

zvec_collection_schema_t* zvec_collection_schema_new(const char* name) {
    auto* schema = new zvec_collection_schema_t;
    schema->ptr = std::make_shared<zvec::CollectionSchema>(std::string(name));
    schema->owned = true;
    return schema;
}

void zvec_collection_schema_free(zvec_collection_schema_t* schema) {
    if (schema && schema->owned) {
        delete schema;
    }
}

zvec_status_t zvec_collection_schema_add_field(zvec_collection_schema_t* schema, zvec_field_schema_t* field) {
    if (!schema || !schema->ptr || !field || !field->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid schema or field pointer");
        return s;
    }
    auto status = schema->ptr->add_field(field->ptr);
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

zvec_status_t zvec_collection_schema_add_index(zvec_collection_schema_t* schema, 
    const char* column_name, zvec_index_params_t* params) {
    if (!schema || !schema->ptr || !column_name || !params || !params->ptr) {
        zvec_status_t s;
        s.code = ZVEC_STATUS_INVALID_ARGUMENT;
        s.message = strdup("Invalid arguments");
        return s;
    }
    auto status = schema->ptr->add_index(std::string(column_name), params->ptr);
    return status.ok() ? zvec_wrapper::ok_status() : zvec_wrapper::to_c_status(status);
}

const char* zvec_collection_schema_name(const zvec_collection_schema_t* schema) {
    if (schema && schema->ptr) {
        return schema->ptr->name().c_str();
    }
    return nullptr;
}

zvec_string_array_t zvec_collection_schema_field_names(const zvec_collection_schema_t* schema) {
    zvec_string_array_t result = {nullptr, 0};
    if (schema && schema->ptr) {
        auto names = schema->ptr->all_field_names();
        result.count = names.size();
        result.strings = (char**)malloc(sizeof(char*) * names.size());
        for (size_t i = 0; i < names.size(); i++) {
            result.strings[i] = strdup(names[i].c_str());
        }
    }
    return result;
}

zvec_string_array_t zvec_collection_schema_vector_field_names(const zvec_collection_schema_t* schema) {
    zvec_string_array_t result = {nullptr, 0};
    if (schema && schema->ptr) {
        auto fields = schema->ptr->vector_fields();
        result.count = fields.size();
        result.strings = (char**)malloc(sizeof(char*) * fields.size());
        for (size_t i = 0; i < fields.size(); i++) {
            result.strings[i] = strdup(fields[i]->name().c_str());
        }
    }
    return result;
}

}
