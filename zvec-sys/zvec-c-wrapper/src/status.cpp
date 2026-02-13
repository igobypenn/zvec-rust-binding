#include "zvec_c_internal.h"
#include <cstring>

extern "C" {

zvec_status_t zvec_status_ok(void) {
    return zvec_wrapper::ok_status();
}

bool zvec_status_is_ok(const zvec_status_t* status) {
    return status && status->code == ZVEC_STATUS_OK;
}

void zvec_status_free(zvec_status_t* status) {
    if (status && status->message) {
        free(const_cast<char*>(status->message));
        status->message = nullptr;
    }
}

void zvec_string_array_free(zvec_string_array_t* arr) {
    if (arr) {
        for (size_t i = 0; i < arr->count; i++) {
            free(arr->strings[i]);
        }
        free(arr->strings);
        arr->strings = nullptr;
        arr->count = 0;
    }
}

}
