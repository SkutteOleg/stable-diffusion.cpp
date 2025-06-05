// Placeholder for SANA DiT, LiteLA, GLUMBConv, etc.
#ifndef SANA_NETS_HPP
#define SANA_NETS_HPP

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <map>
#include <array>

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct gguf_context;


// --- Helper Enums (mirroring Python if applicable) ---
enum class SanaNormType {
    NONE,
    LAYER_NORM,
    RMS_NORM
};

enum class SanaActType {
    NONE,
    SILU,
    GELU
};

// --- Basic Modules (from basic_modules.py and sana_blocks.py) ---

struct SanaRMSNorm {
    ggml_tensor *weight;
    float eps;

    SanaRMSNorm(float eps = 1e-6f) : weight(nullptr), eps(eps) {}
    void init_weights(ggml_context* ctx, ggml_type wtype, int dim);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x);
};

struct SanaLayerNorm {
    ggml_tensor *weight;
    ggml_tensor *bias;
    float eps;
    bool elementwise_affine;

    SanaLayerNorm(float eps = 1e-6f, bool elementwise_affine = true) :
        weight(nullptr), bias(nullptr), eps(eps), elementwise_affine(elementwise_affine) {}
    void init_weights(ggml_context* ctx, ggml_type wtype, int dim);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x);
};

struct SanaGLUMBConv {
    ggml_tensor *inverted_conv_weight;
    ggml_tensor *inverted_conv_bias;
    SanaActType inverted_conv_act_type; // Moved from hpp to ensure it's part of the class

    ggml_tensor *depth_conv_weight;
    ggml_tensor *depth_conv_bias;
    int depth_conv_groups; // Moved from hpp

    ggml_tensor *point_conv_weight;
    ggml_tensor *point_conv_bias;
    SanaActType point_conv_act_type; // Moved from hpp

    SanaActType glu_act_type;
    int C_in, C_hidden, C_out, kernel_size;


    SanaGLUMBConv() :
        inverted_conv_weight(nullptr), inverted_conv_bias(nullptr),
        inverted_conv_act_type(SanaActType::SILU), // Default init
        depth_conv_weight(nullptr), depth_conv_bias(nullptr),
        depth_conv_groups(0), // Default init
        point_conv_weight(nullptr), point_conv_bias(nullptr),
        point_conv_act_type(SanaActType::NONE), // Default init
        glu_act_type(SanaActType::SILU),
        C_in(0), C_hidden(0), C_out(0), kernel_size(3) {}


    SanaGLUMBConv(int c_in, int c_hidden, int c_out, int k_size=3, SanaActType g_act = SanaActType::SILU,
                  SanaActType inv_act = SanaActType::SILU, SanaActType point_act = SanaActType::NONE) :
        inverted_conv_weight(nullptr), inverted_conv_bias(nullptr),
        inverted_conv_act_type(inv_act),
        depth_conv_weight(nullptr), depth_conv_bias(nullptr),
        depth_conv_groups(c_hidden * 2), // Initialize based on c_hidden
        point_conv_weight(nullptr), point_conv_bias(nullptr),
        point_conv_act_type(point_act),
        glu_act_type(g_act), C_in(c_in), C_hidden(c_hidden), C_out(c_out), kernel_size(k_size)
         {}

    void init_weights(ggml_context* ctx, ggml_type wtype);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x, int H, int W);
};

struct SanaLiteLA {
    ggml_tensor *qkv_weight;
    ggml_tensor *qkv_bias;
    ggml_tensor *proj_weight;
    ggml_tensor *proj_bias;

    SanaRMSNorm q_norm;
    SanaRMSNorm k_norm;

    int d_model;
    int num_heads;
    int head_dim;
    float eps_attn;
    bool use_rope;

    SanaLiteLA(int d_model_ = 0, int n_heads = 0, float qk_norm_eps = 1e-5f, float linear_attn_eps = 1e-8f, bool use_rope_ = false) :
        qkv_weight(nullptr), qkv_bias(nullptr), proj_weight(nullptr), proj_bias(nullptr),
        q_norm(qk_norm_eps), k_norm(qk_norm_eps),
        d_model(d_model_), num_heads(n_heads), head_dim(n_heads > 0 ? d_model_ / n_heads : 0),
        eps_attn(linear_attn_eps), use_rope(use_rope_) {}


    void init_weights(ggml_context* ctx, ggml_type wtype);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *image_rotary_emb_cos, ggml_tensor *image_rotary_emb_sin);
};

struct SanaMultiHeadCrossAttention {
    ggml_tensor *q_linear_weight;
    ggml_tensor *q_linear_bias;
    ggml_tensor *kv_linear_weight;
    ggml_tensor *kv_linear_bias;
    ggml_tensor *proj_weight;
    ggml_tensor *proj_bias;

    SanaRMSNorm q_norm;
    SanaRMSNorm k_norm;

    int d_model;
    int num_heads;
    int head_dim;
    int d_cond;

    SanaMultiHeadCrossAttention(int d_model_ = 0, int n_heads = 0, int d_cond_ = 0, bool qk_norm_active = false, float norm_eps = 1e-6f) :
        q_linear_weight(nullptr), q_linear_bias(nullptr), kv_linear_weight(nullptr), kv_linear_bias(nullptr),
        proj_weight(nullptr), proj_bias(nullptr),
        q_norm(norm_eps), k_norm(norm_eps),
        d_model(d_model_), num_heads(n_heads), head_dim(n_heads > 0 ? d_model_ / n_heads : 0), d_cond(d_cond_)
    {
        if (!qk_norm_active) {
            q_norm.weight = nullptr;
            k_norm.weight = nullptr;
        }
    }

    void init_weights(ggml_context* ctx, ggml_type wtype);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *cond, ggml_tensor *mask = nullptr);
};

struct SanaMSBlock {
    SanaLayerNorm norm1;
    SanaLiteLA attn;
    SanaMultiHeadCrossAttention cross_attn;
    SanaLayerNorm norm2;
    SanaGLUMBConv mlp;

    int hidden_size;
    int num_heads;
    float mlp_ratio;

    SanaMSBlock(int h_size, int n_heads, int text_embed_dim, float mlp_r = 4.0f,
                const std::vector<SanaActType>& mlp_acts_config = {SanaActType::SILU, SanaActType::SILU, SanaActType::NONE},
                float qk_norm_eps = 1e-5f, float linear_attn_eps = 1e-8f) :
        norm1(1e-6f, false),
        attn(h_size, n_heads, qk_norm_eps, linear_attn_eps, true),
        cross_attn(h_size, n_heads, text_embed_dim, true, 1e-6f),
        norm2(1e-6f, false),
        mlp(h_size, static_cast<int>(h_size * mlp_r), h_size, 3, mlp_acts_config[1]),
        hidden_size(h_size), num_heads(n_heads), mlp_ratio(mlp_r)
         {
            mlp.inverted_conv_act_type = mlp_acts_config[0];
            mlp.point_conv_act_type = mlp_acts_config[2];
         }

    void init_weights(ggml_context* ctx, ggml_type wtype);
    void load_weights(const std::string& prefix, ggml_context *ctx_gguf, std::map<std::string, ggml_tensor*>& tensors_map);
    ggml_tensor *forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *y, ggml_tensor *t_mod,
                         ggml_tensor *mask, int H_feat, int W_feat,
                         ggml_tensor *image_rotary_emb_cos, ggml_tensor *image_rotary_emb_sin);
};

struct SanaDITModelParams {
    int patch_size = 2;
    int in_channels_vae = 4;
    int hidden_size = 1152;
    int depth = 28;
    int num_heads = 16;
    int out_channels_vae = 4;
    float mlp_ratio = 4.0f;
    bool y_norm_active = true;
    float y_norm_eps = 1e-5f;
    float y_norm_scale_factor = 0.01f;
    int text_embed_dim = 2048;
    int timestep_freq_embed_dim = 256;

    // SANA-Sprint specific
    bool is_sprint_model = false;
    float sprint_sigma_data = 0.5f; // Default from SANA-Sprint config (scheduler.sigma_data)
    bool sprint_cfg_embed = false;  // Corresponds to model.cfg_embed
    float sprint_cfg_embed_scale = 1.0f; // Corresponds to model.cfg_embed_scale
    float sprint_timestep_norm_scale_factor = 1000.0f; // Corresponds to scheduler.timestep_norm_scale_factor
};


struct SanaDITModel {
    SanaDITModelParams params;

    ggml_tensor *x_embedder_conv_w;
    ggml_tensor *x_embedder_conv_b;

    ggml_tensor *t_embedder_mlp_fc1_w;
    ggml_tensor *t_embedder_mlp_fc1_b;
    ggml_tensor *t_embedder_mlp_fc2_w;
    ggml_tensor *t_embedder_mlp_fc2_b;
    ggml_tensor *t_block_linear_w;
    ggml_tensor *t_block_linear_b;

    // Optional: For SANA-Sprint's direct CFG scale embedding
    ggml_tensor *sprint_cfg_embedding_w; // If cfg_embed is a learned embedding

    ggml_tensor *y_proj_fc1_w;
    ggml_tensor *y_proj_fc1_b;
    ggml_tensor *y_proj_fc2_w;
    ggml_tensor *y_proj_fc2_b;
    SanaRMSNorm y_norm;

    std::vector<SanaMSBlock> blocks;

    SanaLayerNorm final_norm;
    ggml_tensor *final_linear_weight;
    ggml_tensor *final_linear_bias;
    ggml_tensor *final_adaLN_modulation_linear_w;
    ggml_tensor *final_adaLN_modulation_linear_b;

    std::map<std::string, ggml_tensor *> tensors_map;


    SanaDITModel(const SanaDITModelParams& p);

    bool load_params_from_gguf(gguf_context *ctx_gguf);
    bool load_weights_from_gguf(ggml_context *ctx_meta, ggml_backend_buffer_t buffer);
    void init_weights(ggml_context *ctx_alloc, ggml_type wtype);


    ggml_cgraph* build_graph(
        ggml_context *ctx_compute,
        ggml_tensor *x_latent,
        ggml_tensor *raw_timestep_embed, // For DiT, this is the direct sinusoidal embedding
                                         // For Sprint, this is the [0, pi/2] SCM timestep value
        ggml_tensor *raw_y_embed,
        ggml_tensor *text_mask,
        ggml_tensor *cfg_scale_value // Tensor containing single float value for CFG scale if using sprint_cfg_embed
    );
};

ggml_tensor* sana_modulate(ggml_context *ctx, ggml_tensor *x, ggml_tensor *shift, ggml_tensor *scale);
ggml_tensor* sana_t2i_modulate(ggml_context *ctx, ggml_tensor *x, ggml_tensor *shift, ggml_tensor *scale);

#define SANA_DIT_GRAPH_SIZE 8192

#endif // SANA_NETS_HPP
