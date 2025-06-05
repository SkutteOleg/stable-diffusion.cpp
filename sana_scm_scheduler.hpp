#ifndef SANA_SCM_SCHEDULER_HPP
#define SANA_SCM_SCHEDULER_HPP

#include "ggml.h"
#include <vector>
#include <string>
#include <cmath>    // For M_PI, atan, etc.
#include <numeric>  // For std::iota
#include <algorithm> // For std::reverse

// Forward declarations
struct ggml_context;
struct ggml_tensor;

struct SanaSCMSchedulerParams {
    int num_train_timesteps = 1000; // Corresponds to num_train_timesteps in Python
    float sigma_data = 0.5f;        // From SANA-Sprint config

    // Parameters for set_timesteps
    float max_timesteps_val = M_PI / 2.0f; // Default (approx 1.57080) from SCM paper / Sana config
    float intermediate_timesteps_val = 1.0f; // Example, from Sana config (might be a list)

    SanaSCMSchedulerParams() = default;
};

struct SanaSCMSchedulerOutput {
    ggml_tensor *prev_sample;
    ggml_tensor *denoised; // x0 prediction
};

class SanaSCMScheduler {
public:
    SanaSCMSchedulerParams params;
    std::vector<float> timesteps; // Discretized timesteps for sampling
    int num_inference_steps = 0;

    SanaSCMScheduler(const SanaSCMSchedulerParams& scheduler_params = SanaSCMSchedulerParams()) : params(scheduler_params) {}

    void set_timesteps(int n_inference_steps,
                       const std::vector<float>* given_timesteps = nullptr,
                       float custom_max_timesteps = -1.0f,       // Allow overriding default max_timesteps_val
                       float custom_intermediate_timesteps = -1.0f // Allow overriding default intermediate
                      ) {
        this->num_inference_steps = n_inference_steps;
        float max_t = (custom_max_timesteps > 0) ? custom_max_timesteps : params.max_timesteps_val;

        if (given_timesteps && !given_timesteps->empty()) {
            this->timesteps = *given_timesteps;
        } else {
            this->timesteps.resize(n_inference_steps + 1);
            if (n_inference_steps == 1) { // One-step generation
                this->timesteps[0] = max_t;
                this->timesteps[1] = 0.0f;
            } else if (n_inference_steps == 2 && (custom_intermediate_timesteps > 0 || params.intermediate_timesteps_val > 0) ) {
                float inter_t = (custom_intermediate_timesteps > 0) ? custom_intermediate_timesteps : params.intermediate_timesteps_val;
                this->timesteps[0] = max_t;
                this->timesteps[1] = inter_t;
                this->timesteps[2] = 0.0f;
            }
            else { // Linspace for multi-step
                for (int i = 0; i <= n_inference_steps; ++i) {
                    this->timesteps[i] = max_t - (max_t / n_inference_steps) * i;
                }
            }
        }
        // Ensure timesteps are float32 for ggml
        // (already float, just conceptual note)
    }

    // Corresponds to the 'trigflow' parameterization step in SCMScheduler.py
    SanaSCMSchedulerOutput step(
        ggml_context* ctx,
        ggml_tensor* model_output, // Flow model output (velocity v_t or data prediction x_theta)
        int time_index,            // Current index in the timesteps vector
        ggml_tensor* sample        // Current noisy sample x_s (or x_t in paper)
                                   // Note: Python's `timestep` is s, `t` is the next step in loop.
                                   // Here, `time_index` refers to current step `s`.
    );
};

#endif // SANA_SCM_SCHEDULER_HPP
