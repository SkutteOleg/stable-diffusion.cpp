#include "sana_dc_ae.hpp"
#include "gguf.h"      // For GGUF context and functions
#include <cstdio>      // For snprintf
#include <vector>      // For std::vector if used in more complex blocks
#include <stdexcept>   // For std::runtime_error
#include <map>         // For tensors_map in load_weights

// Helper to get tensor from map (similar to one in sana_text_encoder.cpp)
// This could be moved to a shared utility header if used in multiple places.
static ggml_tensor* get_tensor_from_map_or_fail_dcae(const std::map<std::string, ggml_tensor*>& tensors_map, const std::string& name) {
    auto it = tensors_map.find(name);
    if (it == tensors_map.end()) {
        throw std::runtime_error("SanaDCAE: Tensor not found in map: " + name);
    }
    return it->second;
}
static ggml_tensor* get_tensor_from_map_optional_dcae(const std::map<std::string, ggml_tensor*>& tensors_map, const std::string& name) {
    auto it = tensors_map.find(name);
    if (it == tensors_map.end()) {
        return nullptr;
    }
    return it->second;
}


// --- SanaDCAE ---

SanaDCAE::SanaDCAE(const SanaDCAEConfig& cfg) : config(cfg) {
    enc_conv_in_weight = nullptr;
    enc_conv_in_bias = nullptr;
    // dec_conv_out_weight = nullptr; // etc.
}

bool SanaDCAE::load_params_from_gguf(gguf_context *ctx_gguf) {
    const char* prefix = "sana_dcae."; // Example prefix
    std::string key;

    auto get_u32_optional = [&](const char* suffix, int& val, int default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_UINT32)
            val = gguf_get_val_u32(ctx_gguf, k_idx);
        else if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_INT32) // Allow int32 as well
            val = gguf_get_val_i32(ctx_gguf, k_idx);
        else val = default_val;
    };
    auto get_f32_optional = [&](const char* suffix, float& val, float default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_FLOAT32) val = gguf_get_val_f32(ctx_gguf, k_idx);
        else val = default_val;
    };
    auto get_arr_int_optional = [&](const char* suffix, std::array<int,4>& arr, const std::array<int,4>& default_arr) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx_gguf, k_idx) == GGUF_TYPE_INT32 && gguf_get_arr_n(ctx_gguf, k_idx) == 4) {
            const int32_t* data = (const int32_t*)gguf_get_arr_data(ctx_gguf, k_idx);
            for(int i=0; i<4; ++i) arr[i] = data[i];
        } else {
            arr = default_arr;
        }
    };

    get_u32_optional("vae_latent_dim", config.vae_latent_dim, 32);
    get_u32_optional("vae_downsample_rate", config.vae_downsample_rate, 32);
    get_f32_optional("scaling_factor", config.scaling_factor, 0.18215f); // Common SD scaling factor as a fallback
    get_u32_optional("image_channels", config.image_channels, 3);

    std::array<int, 4> default_enc_blocks = {128, 256, 512, 512};
    std::array<int, 4> default_dec_blocks = {512, 512, 256, 128};
    get_arr_int_optional("encoder.block_out_channels", config.encoder_block_out_channels, default_enc_blocks);
    get_arr_int_optional("decoder.block_in_channels", config.decoder_block_in_channels, default_dec_blocks);

    // If vae_downsample_rate was not in GGUF, calculate it based on encoder blocks if possible
    // This is a heuristic. A fixed value from GGUF is better.
    if (gguf_find_key(ctx_gguf, (std::string(prefix) + "vae_downsample_rate").c_str()) == -1 ) {
        config.vae_downsample_rate = 1;
        for(size_t i=0; i < config.encoder_block_out_channels.size(); ++i) { // Assuming each block implies a /2 downsample
            if (i > 0 && config.encoder_block_out_channels[i] > config.encoder_block_out_channels[i-1]) { // Heuristic for downsampling stage
                 config.vae_downsample_rate *= 2;
            } else if (i==0 && config.encoder_block_out_channels[0] > config.image_channels*4) { // Heuristic for initial conv stride
                 config.vae_downsample_rate *=2;
            }
        }
        if (config.vae_downsample_rate < 4) config.vae_downsample_rate = 8; // Minimum sensible default
        fprintf(stdout, "SanaDCAE: Guessed vae_downsample_rate: %d based on encoder_block_out_channels.\n", config.vae_downsample_rate);
    }


    return true; // Indicate params are loaded/defaulted
}


bool SanaDCAE::load_weights_from_gguf(ggml_context *ctx_meta, ggml_backend_buffer_t buffer, std::map<std::string, struct ggml_tensor *>& model_tensors_map) {
    (void)buffer; // Not using buffer directly for allocation in this simplified version
    char name_buf[256];
    const char* prefix = "sana_dcae."; // Example prefix for VAE weights

    auto load_tensor = [&](const char* suffix, ggml_tensor*& target_tensor, bool required = true) {
        snprintf(name_buf, sizeof(name_buf), "%s%s", prefix, suffix);
        target_tensor = ggml_get_tensor(ctx_meta, name_buf); // Get metadata pointer
        if (target_tensor) {
            model_tensors_map[name_buf] = target_tensor; // Store in the main model's map
        } else if (required) {
            fprintf(stderr, "SanaDCAE: Required tensor %s not found.\n", name_buf);
            return false;
        } else {
            // fprintf(stderr, "SanaDCAE: Optional tensor %s not found.\n", name_buf);
        }
        return true;
    };

    // Example: Load a few key tensors for the simplified VAE structure
    if (!load_tensor("encoder.conv_in.weight", enc_conv_in_weight)) return false;
    load_tensor("encoder.conv_in.bias", enc_conv_in_bias, false);
    // ... load other encoder block weights based on actual GGUF names ...
    // if (!load_tensor("decoder.conv_out.weight", dec_conv_out_weight)) return false;
    // load_tensor("decoder.conv_out.bias", dec_conv_out_bias, false);

    return true;
}


ggml_tensor* SanaDCAE::encode(ggml_context *ctx, ggml_tensor *image) {
    // image: (W, H, C_in, B)
    // Output: (W_latent, H_latent, C_latent, B)

    ggml_tensor* current = image;
    if (enc_conv_in_weight) { // Basic first conv
        current = ggml_conv_2d_sk_p0(ctx, enc_conv_in_weight, current);
        if (enc_conv_in_bias) {
             current = ggml_add_inplace(ctx, current, ggml_reshape_4d(ctx, enc_conv_in_bias, 1, 1, current->ne[2], current->ne[3]));
        }
        current = ggml_relu_inplace(ctx, current);
    }

    int W_latent = image->ne[0] / config.vae_downsample_rate;
    int H_latent = image->ne[1] / config.vae_downsample_rate;
    if (W_latent <= 0) W_latent = 1;
    if (H_latent <= 0) H_latent = 1;

    // Placeholder: In a real VAE, `current` would be processed through more layers
    // and its shape would naturally become W_latent, H_latent, config.vae_latent_dim
    // For now, we create a new tensor of the target shape if current isn't already it.
    // This is a significant simplification.
    ggml_tensor *latent_placeholder = ggml_new_tensor_4d(ctx, current->type, W_latent, H_latent, config.vae_latent_dim, image->ne[3]);
    ggml_set_name(latent_placeholder, "encoded_latent_placeholder");
    // Here one might copy or reshape `current` into `latent_placeholder` if dimensions allow

    latent_placeholder = ggml_scale_inplace(ctx, latent_placeholder, config.scaling_factor);
    ggml_set_name(latent_placeholder, "scaled_encoded_latent");

    return latent_placeholder;
}

ggml_tensor* SanaDCAE::decode(ggml_context *ctx, ggml_tensor *latent) {
    ggml_tensor* current = ggml_scale(ctx, latent, 1.0f / config.scaling_factor);
    ggml_set_name(current, "scaled_latent_for_decode");

    // Placeholder for decoder
    // Example: Initial upsampling conv (conceptual)
    // if (dec_conv_in_weight) {
    //      current = ggml_conv_transpose_2d(ctx, dec_conv_in_weight, current, ...);
    //      if(dec_conv_in_bias) ...
    //      current = ggml_relu_inplace(ctx, current);
    // }
    // ... more upsampling blocks ...
    // if (dec_conv_out_weight) { // Final conv to image channels
    //      current = ggml_conv_transpose_2d(ctx, dec_conv_out_weight, current, ...);
    //      if(dec_conv_out_bias) ...
    // }
    // current = ggml_sigmoid_inplace(ctx, current); // Or tanh


    int W_out = current->ne[0] * config.vae_downsample_rate;
    int H_out = current->ne[1] * config.vae_downsample_rate;
    if (W_out <= 0) W_out = config.vae_downsample_rate;
    if (H_out <= 0) H_out = config.vae_downsample_rate;

    ggml_tensor *image_placeholder = ggml_new_tensor_4d(ctx, latent->type, W_out, H_out, config.image_channels, latent->ne[3]);
    ggml_set_name(image_placeholder, "decoded_image_placeholder");

    return image_placeholder;
}

void SanaDCAE::init_weights(ggml_context *ctx_w, ggml_type wtype, int H_in, int W_in) {
    (void)H_in; (void)W_in; // Unused for now with simplified structure
    char name_buf[128];

    // Example for enc_conv_in
    snprintf(name_buf, sizeof(name_buf), "sana_dcae.encoder.conv_in.weight");
    enc_conv_in_weight = ggml_new_tensor_4d(ctx_w, wtype, 3, 3, config.image_channels, config.encoder_block_out_channels[0]);
    ggml_set_name(enc_conv_in_weight, name_buf);

    snprintf(name_buf, sizeof(name_buf), "sana_dcae.encoder.conv_in.bias");
    enc_conv_in_bias = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, config.encoder_block_out_channels[0]);
    ggml_set_name(enc_conv_in_bias, name_buf);

    // Initialize other conceptual layers if they were defined in the struct
    // e.g. dec_conv_out_weight, etc.
}
