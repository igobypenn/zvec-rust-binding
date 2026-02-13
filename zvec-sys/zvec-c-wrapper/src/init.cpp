#include "zvec_c_internal.h"
#include <zvec/core/framework/index_factory.h>
#include <vector>
#include <string>

extern "C" {

bool zvec_init(void) {
    auto metrics = zvec::core::IndexFactory::AllMetrics();
    return !metrics.empty();
}

int zvec_list_registered_metrics(const char*** out_metrics) {
    static std::vector<std::string> cached_metrics;
    static std::vector<const char*> cached_ptrs;
    
    if (cached_metrics.empty()) {
        cached_metrics = zvec::core::IndexFactory::AllMetrics();
        cached_ptrs.reserve(cached_metrics.size());
        for (const auto& m : cached_metrics) {
            cached_ptrs.push_back(m.c_str());
        }
    }
    
    *out_metrics = cached_ptrs.data();
    return static_cast<int>(cached_ptrs.size());
}

int zvec_list_registered_builders(const char*** out_builders) {
    static std::vector<std::string> cached_builders;
    static std::vector<const char*> cached_ptrs;
    
    if (cached_builders.empty()) {
        cached_builders = zvec::core::IndexFactory::AllBuilders();
        cached_ptrs.reserve(cached_builders.size());
        for (const auto& b : cached_builders) {
            cached_ptrs.push_back(b.c_str());
        }
    }
    
    *out_builders = cached_ptrs.data();
    return static_cast<int>(cached_ptrs.size());
}

int zvec_list_registered_searchers(const char*** out_searchers) {
    static std::vector<std::string> cached_searchers;
    static std::vector<const char*> cached_ptrs;
    
    if (cached_searchers.empty()) {
        cached_searchers = zvec::core::IndexFactory::AllSearchers();
        cached_ptrs.reserve(cached_searchers.size());
        for (const auto& s : cached_searchers) {
            cached_ptrs.push_back(s.c_str());
        }
    }
    
    *out_searchers = cached_ptrs.data();
    return static_cast<int>(cached_ptrs.size());
}

int zvec_list_registered_streamers(const char*** out_streamers) {
    static std::vector<std::string> cached_streamers;
    static std::vector<const char*> cached_ptrs;
    
    if (cached_streamers.empty()) {
        cached_streamers = zvec::core::IndexFactory::AllStreamers();
        cached_ptrs.reserve(cached_streamers.size());
        for (const auto& s : cached_streamers) {
            cached_ptrs.push_back(s.c_str());
        }
    }
    
    *out_streamers = cached_ptrs.data();
    return static_cast<int>(cached_ptrs.size());
}

}
