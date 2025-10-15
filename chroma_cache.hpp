#pragma once

#include "model.h"
#include <map>
#include <string>
#include <vector>
#include "ggml_extend.hpp"

struct CacheContext {
    std::map<std::string, ggml_tensor*> buffers;
    int sequence_num           = 0;
    bool use_cache             = false;
    int consecutive_cache_hits = 0;

    ggml_tensor* get_buffer(const std::string& name) {
        auto it = buffers.find(name);
        if (it != buffers.end()) {
            return it->second;
        }
        return nullptr;
    }

    void set_buffer(const std::string& name, ggml_tensor* buffer) {
        buffers[name] = buffer;
    }

    void clear_buffers() {
        sequence_num = 0;
        for (auto const& [key, val] : buffers) {
            // we don't manage memory here, just clear the map
        }
        buffers.clear();
    }
};

static thread_local CacheContext* current_cache_context = nullptr;

static float chroma_cache_start_sigma = 0.f;
static float chroma_cache_end_sigma   = 0.f;
static int chroma_cache_interval      = 0;

CacheContext* get_current_cache_context() {
    return current_cache_context;
}

void set_current_cache_context(CacheContext* cache_context) {
    current_cache_context = cache_context;
}

static CacheContext global_cache_context;

void ensure_cache_context() {
    if (get_current_cache_context() == nullptr) {
        set_current_cache_context(&global_cache_context);
    }
}

void reset_cache_state() {
    if (current_cache_context != nullptr) {
        current_cache_context->clear_buffers();
        current_cache_context->consecutive_cache_hits = 0;
    }
}