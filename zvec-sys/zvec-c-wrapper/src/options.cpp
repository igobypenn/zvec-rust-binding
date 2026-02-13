#include "zvec_c_internal.h"

extern "C" {

zvec_collection_options_t* zvec_collection_options_new(void) {
    return new zvec_collection_options_t;
}

void zvec_collection_options_free(zvec_collection_options_t* options) {
    delete options;
}

void zvec_collection_options_set_read_only(zvec_collection_options_t* options, bool read_only) {
    if (options) {
        options->opts.read_only_ = read_only;
    }
}

void zvec_collection_options_set_enable_mmap(zvec_collection_options_t* options, bool enable_mmap) {
    if (options) {
        options->opts.enable_mmap_ = enable_mmap;
    }
}

void zvec_collection_options_set_max_buffer_size(zvec_collection_options_t* options, uint64_t max_buffer_size) {
    if (options) {
        options->opts.max_buffer_size_ = static_cast<uint32_t>(max_buffer_size);
    }
}

zvec_create_index_options_t* zvec_create_index_options_new(void) {
    return new zvec_create_index_options_t;
}

void zvec_create_index_options_free(zvec_create_index_options_t* options) {
    delete options;
}

void zvec_create_index_options_set_concurrency(zvec_create_index_options_t* options, int concurrency) {
    if (options) {
        options->opts.concurrency_ = concurrency;
    }
}

zvec_optimize_options_t* zvec_optimize_options_new(void) {
    return new zvec_optimize_options_t;
}

void zvec_optimize_options_free(zvec_optimize_options_t* options) {
    delete options;
}

void zvec_optimize_options_set_concurrency(zvec_optimize_options_t* options, int concurrency) {
    if (options) {
        options->opts.concurrency_ = concurrency;
    }
}

}
