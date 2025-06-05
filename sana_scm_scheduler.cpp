#include "sana_scm_scheduler.hpp"
#include "ggml.h"
#include <vector>
#include <cmath> // For cosf, sinf

SanaSCMSchedulerOutput SanaSCMScheduler::step(
    ggml_context* ctx,
    ggml_tensor* model_output_flow,
    int time_index,
    ggml_tensor* sample
) {
    if (num_inference_steps == 0 || timesteps.empty()) {
        // Consider logging an error or returning a more indicative error state
        return {nullptr, nullptr};
    }

    // Ensure time_index is valid
    if (time_index < 0 || (size_t)time_index >= timesteps.size()) {
         // Consider logging an error
        return {nullptr, nullptr};
    }

    float s_val = timesteps[time_index];
    float t_val = (time_index + 1 < (int)timesteps.size()) ? timesteps[time_index + 1] : 0.0f;

    ggml_tensor* s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float *)s->data)[0] = s_val;
    ggml_set_name(s, "s_val_tensor");

    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float *)t->data)[0] = t_val;
    ggml_set_name(t, "t_val_tensor");

    ggml_tensor* cos_s = ggml_cos(ctx, s);
    ggml_tensor* sin_s = ggml_sin(ctx, s);

    // pred_x0 = cos(s) * sample - sin(s) * model_output_flow
    // Need to ensure tensors are broadcastable. If sample/model_output_flow are (C,N,B) and cos_s/sin_s are scalar-like (1) or (1,1,1)
    ggml_tensor* term1_pred_x0 = ggml_mul(ctx, ggml_repeat(ctx, cos_s, sample), sample);
    ggml_tensor* term2_pred_x0 = ggml_mul(ctx, ggml_repeat(ctx, sin_s, model_output_flow), model_output_flow);
    ggml_tensor* pred_x0 = ggml_sub(ctx, term1_pred_x0, term2_pred_x0);
    ggml_set_name(pred_x0, "pred_x0_scm");

    ggml_tensor* prev_sample = nullptr;
    // Check if it's not the last step in the timesteps array (which would mean t_val is the final target time, often 0)
    if (time_index + 1 < (int)timesteps.size()) {
        ggml_tensor* cos_t = ggml_cos(ctx, t);
        // ggml_tensor* sin_t = ggml_sin(ctx, t); // Not used in this deterministic step version

        // prev_sample = cos(t) * pred_x0
        // (In a stochastic sampler, it would be + sin(t) * noise)
        prev_sample = ggml_mul(ctx, ggml_repeat(ctx, cos_t, pred_x0), pred_x0);
        ggml_set_name(prev_sample, "prev_sample_scm");

    } else { // This is the last step, so prev_sample is effectively the denoised x0
        prev_sample = ggml_dup(ctx, pred_x0);
        ggml_set_name(prev_sample, "final_denoised_sample_scm");
    }

    return {prev_sample, pred_x0};
}
