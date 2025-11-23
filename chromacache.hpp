#pragma once

#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "denoiser.hpp"
#include "ggml_extend.hpp"

struct ChromaCacheConfig {
    bool enabled          = false;
    float start_percent   = 0.0f;
    float end_percent     = 0.0f;
    int interval          = 0;
};

struct ChromaCacheState {
    ChromaCacheConfig config;
    Denoiser* denoiser     = nullptr;
    float start_sigma      = 0.0f;
    float end_sigma        = 0.0f;
    bool initialized       = false;
    bool use_cache         = false;
    int consecutive_cache_hits = 0;

    std::map<std::string, ggml_tensor*> buffers;

    void init(const ChromaCacheConfig& cfg, Denoiser* d) {
        config      = cfg;
        denoiser    = d;
        initialized = cfg.enabled && d != nullptr;
        reset_runtime();
        if (initialized) {
            start_sigma = percent_to_sigma(config.start_percent);
            end_sigma   = percent_to_sigma(config.end_percent);
        }
    }

    void reset_runtime() {
        buffers.clear();
        consecutive_cache_hits = 0;
        use_cache              = false;
    }

    bool enabled() const {
        return initialized && config.enabled;
    }

    float percent_to_sigma(float percent) const {
        if (!denoiser) {
            return 0.0f;
        }
        if (percent <= 0.0f) {
            return std::numeric_limits<float>::max();
        }
        if (percent >= 1.0f) {
            return 0.0f;
        }
        float t = (1.0f - percent) * (TIMESTEPS - 1);
        return denoiser->t_to_sigma(t);
    }

    bool check_cache(float sigma) {
        if (!enabled()) {
            use_cache = false;
            return false;
        }
        
        bool can_use_cache = buffers.find("hidden_states") != buffers.end();
        can_use_cache      = can_use_cache && sigma <= start_sigma && sigma >= end_sigma;
        can_use_cache      = can_use_cache && consecutive_cache_hits < config.interval;
        
        if (can_use_cache) {
            consecutive_cache_hits++;
        } else {
            consecutive_cache_hits = 0;
        }
        use_cache = can_use_cache;
        return use_cache;
    }

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
};
