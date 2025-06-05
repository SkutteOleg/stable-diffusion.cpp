#include "sana_nets.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <queue>
#include <regex>

#ifndef SANA_TEXT_ENCODER_GRAPH_SIZE
#define SANA_TEXT_ENCODER_GRAPH_SIZE 4096
#endif

// --- Unicode utilities (simplified from llama.cpp) ---
static size_t utf8_len_char(const char *s, size_t max_len) {
    if (max_len == 0) return 0;
    const unsigned char c = (unsigned char)(*s);
    if (c < 0x80) return 1;
    if (max_len >= 2 && (c & 0xE0) == 0xC0) return 2;
    if (max_len >= 3 && (c & 0xF0) == 0xE0) return 3;
    if (max_len >= 4 && (c & 0xF8) == 0xF0) return 4;
    return 0;
}

// --- Static Helper Functions for Tensor Map Access ---
static ggml_tensor* get_tensor_from_map_or_fail(const std::map<std::string, ggml_tensor*>& tensors_map, const std::string& name) {
    auto it = tensors_map.find(name);
    if (it == tensors_map.end()) {
        throw std::runtime_error("SanaNet: Tensor not found in map: " + name);
    }
    return it->second;
}

static ggml_tensor* get_tensor_from_map_optional(const std::map<std::string, ggml_tensor*>& tensors_map, const std::string& name) {
    auto it = tensors_map.find(name);
    if (it == tensors_map.end()) {
        return nullptr;
    }
    return it->second;
}


// --- Helper Function Implementations ---
ggml_tensor* sana_modulate(ggml_context *ctx, ggml_tensor *x, ggml_tensor *shift, ggml_tensor *scale) {
    ggml_tensor* scale_reshaped = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
    ggml_tensor* shift_reshaped = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);

    ggml_tensor* one_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float *)one_val->data)[0] = 1.0f;
    ggml_tensor* one_reshaped = ggml_reshape_3d(ctx, one_val, 1, 1, 1);

    ggml_tensor* term_scale = ggml_add(ctx, one_reshaped, scale_reshaped);

    ggml_tensor* x_out = ggml_mul(ctx, x, term_scale);
    x_out = ggml_add(ctx, x_out, shift_reshaped);
    return x_out;
}

ggml_tensor* sana_t2i_modulate(ggml_context *ctx, ggml_tensor *x, ggml_tensor *shift, ggml_tensor *scale) {
    ggml_tensor* one_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float *)one_val->data)[0] = 1.0f;
    ggml_tensor* one_reshaped = ggml_reshape_3d(ctx, one_val, 1, 1, 1);

    ggml_tensor* scale_plus_one = ggml_add(ctx, one_reshaped, scale);
    ggml_tensor* x_scaled = ggml_mul(ctx, x, scale_plus_one);
    ggml_tensor* x_shifted = ggml_add(ctx, x_scaled, shift);
    return x_shifted;
}

// --- SanaRMSNorm ---
void SanaRMSNorm::init_weights(ggml_context* ctx, ggml_type wtype, int dim) {
    weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
}
void SanaRMSNorm::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    (void)ctx_meta;
    weight = get_tensor_from_map_optional(tensors_map_in, prefix + ".weight");
}
ggml_tensor *SanaRMSNorm::forward(ggml_context *ctx, ggml_tensor *x) {
    ggml_tensor * out = ggml_rms_norm(ctx, x, eps);
    if (weight) {
        ggml_tensor* reshaped_weight = ggml_reshape_3d(ctx, weight, weight->ne[0], 1, 1);
        out = ggml_mul(ctx, out, reshaped_weight);
    }
    return out;
}

// --- SanaLayerNorm ---
void SanaLayerNorm::init_weights(ggml_context* ctx, ggml_type wtype, int dim) {
    if (elementwise_affine) {
        weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    }
}
void SanaLayerNorm::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    (void)ctx_meta;
    if (elementwise_affine) {
        weight = get_tensor_from_map_optional(tensors_map_in, prefix + ".weight");
        bias = get_tensor_from_map_optional(tensors_map_in, prefix + ".bias");
    }
}
ggml_tensor *SanaLayerNorm::forward(ggml_context *ctx, ggml_tensor *x) {
    ggml_tensor * out = ggml_norm(ctx, x, eps);
    if (elementwise_affine && weight) {
        ggml_tensor* reshaped_weight = ggml_reshape_3d(ctx, weight, weight->ne[0], 1, 1);
        out = ggml_mul(ctx, out, reshaped_weight);
        if (bias) {
            ggml_tensor* reshaped_bias = ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1);
            out = ggml_add(ctx, out, reshaped_bias);
        }
    }
    return out;
}

// --- SanaGLUMBConv ---
void SanaGLUMBConv::init_weights(ggml_context* ctx, ggml_type wtype) {
    inverted_conv_weight = ggml_new_tensor_4d(ctx, wtype, 1, 1, this->C_in, this->C_hidden * 2);
    inverted_conv_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, this->C_hidden * 2);
    depth_conv_weight    = ggml_new_tensor_4d(ctx, wtype, this->kernel_size, this->kernel_size, 1, this->C_hidden * 2);
    //this->depth_conv_groups is already initialized in constructor
    depth_conv_bias      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, this->C_hidden * 2);
    point_conv_weight    = ggml_new_tensor_4d(ctx, wtype, 1, 1, this->C_hidden, this->C_out);
    point_conv_bias      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, this->C_out);
}
void SanaGLUMBConv::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    (void)ctx_meta;
    inverted_conv_weight = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".inverted_conv.weight");
    inverted_conv_bias   = get_tensor_from_map_optional(tensors_map_in, prefix + ".inverted_conv.bias");
    depth_conv_weight    = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".depth_conv.weight");
    depth_conv_bias      = get_tensor_from_map_optional(tensors_map_in, prefix + ".depth_conv.bias");
    point_conv_weight    = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".point_conv.weight");
    point_conv_bias      = get_tensor_from_map_optional(tensors_map_in, prefix + ".point_conv.bias");

    this->C_in = inverted_conv_weight->ne[2];
    this->C_hidden = point_conv_weight->ne[2];
    this->C_out = point_conv_weight->ne[3];
    this->kernel_size = depth_conv_weight->ne[0];
    this->depth_conv_groups = inverted_conv_weight->ne[3];
}
ggml_tensor* SanaGLUMBConv::forward(ggml_context *ctx, ggml_tensor *x_in_3d, int H, int W) {
    const int64_t C_in_runtime = x_in_3d->ne[0];
    const int64_t B_in = x_in_3d->ne[2];

    ggml_tensor* x = ggml_reshape_4d(ctx, x_in_3d, W, H, C_in_runtime, B_in);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 2, 3));

    ggml_tensor *hidden = ggml_conv_2d(ctx, this->inverted_conv_weight, x, 1, 1, 0, 0, 1, 1);
    if (this->inverted_conv_bias) {
        hidden = ggml_add_inplace(ctx, hidden, ggml_reshape_4d(ctx, this->inverted_conv_bias, 1, 1, hidden->ne[2], B_in));
    }
    if (this->inverted_conv_act_type == SanaActType::SILU) {
        hidden = ggml_silu_inplace(ctx, hidden);
    }

    ggml_tensor *depth_out = ggml_conv_2d(ctx, this->depth_conv_weight, hidden, 1, 1, this->kernel_size/2, this->kernel_size/2, 1, this->depth_conv_groups);
    if (this->depth_conv_bias) {
         depth_out = ggml_add_inplace(ctx, depth_out, ggml_reshape_4d(ctx, this->depth_conv_bias, 1, 1, depth_out->ne[2], B_in));
    }

    int C_hidden_x2_runtime = depth_out->ne[2];
    ggml_tensor *val    = ggml_view_4d(ctx, depth_out, W, H, C_hidden_x2_runtime/2, B_in, depth_out->nb[1], depth_out->nb[2], depth_out->nb[3], 0);
    ggml_tensor *gate_val = ggml_view_4d(ctx, depth_out, W, H, C_hidden_x2_runtime/2, B_in, depth_out->nb[1], depth_out->nb[2], depth_out->nb[3], (C_hidden_x2_runtime/2)*ggml_element_size(depth_out));

    if (this->glu_act_type == SanaActType::SILU) {
        gate_val = ggml_silu_inplace(ctx, gate_val);
    }
    x = ggml_mul_inplace(ctx, val, gate_val);

    x = ggml_conv_2d(ctx, this->point_conv_weight, x, 1, 1, 0, 0, 1, 1);
    if (this->point_conv_bias) {
        x = ggml_add_inplace(ctx, x, ggml_reshape_4d(ctx, this->point_conv_bias, 1, 1, x->ne[2], B_in));
    }
    if (this->point_conv_act_type == SanaActType::SILU) {
        x = ggml_silu_inplace(ctx, x);
    }

    x = ggml_permute(ctx, x, 2, 0, 1, 3);
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1]*x->ne[2], x->ne[3]);
    return x;
}

// --- SanaLiteLA ---
void SanaLiteLA::init_weights(ggml_context* ctx, ggml_type wtype) {
    qkv_weight = ggml_new_tensor_2d(ctx, wtype, d_model * 3, d_model);
    qkv_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model * 3);
    proj_weight = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
    proj_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
    q_norm.init_weights(ctx, wtype, d_model);
    k_norm.init_weights(ctx, wtype, d_model);
}
void SanaLiteLA::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    (void)ctx_meta;
    qkv_weight = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".qkv.weight");
    qkv_bias   = get_tensor_from_map_optional(tensors_map_in, prefix + ".qkv.bias");
    proj_weight= get_tensor_from_map_or_fail(tensors_map_in, prefix + ".proj.weight");
    proj_bias  = get_tensor_from_map_optional(tensors_map_in, prefix + ".proj.bias");
    q_norm.load_weights(prefix + ".q_norm", ctx_meta, tensors_map_in);
    k_norm.load_weights(prefix + ".k_norm", ctx_meta, tensors_map_in);
    this->d_model = qkv_weight->ne[1];
    if(this->num_heads > 0) this->head_dim = this->d_model / this->num_heads; else if (this->d_model > 0) this->head_dim = this->d_model;
}
ggml_tensor* SanaLiteLA::forward(ggml_context *ctx, ggml_tensor *x_in_3d, ggml_tensor *image_rotary_emb_cos, ggml_tensor *image_rotary_emb_sin) {
    const int64_t C = x_in_3d->ne[0];
    const int64_t N = x_in_3d->ne[1];
    const int64_t B = x_in_3d->ne[2];

    ggml_tensor *qkv = ggml_mul_mat(ctx, this->qkv_weight, x_in_3d);
    if (this->qkv_bias) {
        qkv = ggml_add_inplace(ctx, qkv, ggml_reshape_3d(ctx, this->qkv_bias, this->qkv_bias->ne[0], 1, 1));
    }

    ggml_tensor *q = ggml_view_3d(ctx, qkv, C, N, B, qkv->nb[1], qkv->nb[2], 0);
    ggml_tensor *k = ggml_view_3d(ctx, qkv, C, N, B, qkv->nb[1], qkv->nb[2], C * ggml_element_size(qkv));
    ggml_tensor *v = ggml_view_3d(ctx, qkv, C, N, B, qkv->nb[1], qkv->nb[2], 2 * C * ggml_element_size(qkv));

    q = this->q_norm.forward(ctx, q);
    k = this->k_norm.forward(ctx, k);

    q = ggml_reshape_4d(ctx, q, this->head_dim, this->num_heads, N, B);
    k = ggml_reshape_4d(ctx, k, this->head_dim, this->num_heads, N, B);
    v = ggml_reshape_4d(ctx, v, this->head_dim, this->num_heads, N, B);

    q = ggml_relu(ctx, q);
    k = ggml_relu(ctx, k);

    ggml_tensor *q_fa = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor *k_fa = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor *v_fa = ggml_permute(ctx, v, 0, 2, 1, 3);

    ggml_tensor *out_attn = ggml_flash_attn_ext(ctx, q_fa, k_fa, v_fa, nullptr, 1.0f, 0.0f, 0 );
    out_attn = ggml_permute(ctx, out_attn, 0, 2, 1, 3);

    out_attn = ggml_reshape_3d(ctx, out_attn, C, N, B);

    ggml_tensor *out = ggml_mul_mat(ctx, this->proj_weight, out_attn);
    if (this->proj_bias) {
        out = ggml_add_inplace(ctx, out, ggml_reshape_3d(ctx, this->proj_bias, C,1,1));
    }
    return out;
}

// --- SanaMultiHeadCrossAttention ---
void SanaMultiHeadCrossAttention::init_weights(ggml_context* ctx, ggml_type wtype) {
    q_linear_weight = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
    kv_linear_weight = ggml_new_tensor_2d(ctx, wtype, d_cond, d_model * 2);
    proj_weight = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
    q_norm.init_weights(ctx, wtype, d_model);
    k_norm.init_weights(ctx, wtype, d_model);
}
void SanaMultiHeadCrossAttention::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    (void)ctx_meta;
    q_linear_weight = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".q_linear.weight");
    kv_linear_weight= get_tensor_from_map_or_fail(tensors_map_in, prefix + ".kv_linear.weight");
    proj_weight     = get_tensor_from_map_or_fail(tensors_map_in, prefix + ".proj.weight");
    q_norm.load_weights(prefix + ".q_norm", ctx_meta, tensors_map_in);
    k_norm.load_weights(prefix + ".k_norm", ctx_meta, tensors_map_in);
    this->d_model = q_linear_weight->ne[1];
    this->d_cond = kv_linear_weight->ne[1];
    if(this->num_heads > 0) this->head_dim = this->d_model / this->num_heads; else if (this->d_model > 0) this->head_dim = this->d_model;
}
ggml_tensor* SanaMultiHeadCrossAttention::forward(ggml_context *ctx, ggml_tensor *x, ggml_tensor *cond, ggml_tensor *mask) {
    const int64_t C = x->ne[0];
    const int64_t N_q = x->ne[1];
    const int64_t B = x->ne[2];

    const int64_t C_kv = cond->ne[0];
    const int64_t N_kv = cond->ne[1];

    ggml_tensor *q_proj = ggml_mul_mat(ctx, this->q_linear_weight, x);
    if (this->q_linear_bias) q_proj = ggml_add_inplace(ctx, q_proj, ggml_reshape_3d(ctx, this->q_linear_bias, C,1,1));

    ggml_tensor *kv_proj = ggml_mul_mat(ctx, this->kv_linear_weight, cond);
    if (this->kv_linear_bias) kv_proj = ggml_add_inplace(ctx, kv_proj,  ggml_reshape_3d(ctx, this->kv_linear_bias, this->kv_linear_bias->ne[0],1,1));

    ggml_tensor *k_proj = ggml_view_3d(ctx, kv_proj, C_kv, N_kv, B, kv_proj->nb[1], kv_proj->nb[2], 0);
    ggml_tensor *v_proj = ggml_view_3d(ctx, kv_proj, C_kv, N_kv, B, kv_proj->nb[1], kv_proj->nb[2], C_kv * ggml_element_size(kv_proj));

    q_proj = this->q_norm.forward(ctx, q_proj);
    k_proj = this->k_norm.forward(ctx, k_proj);

    ggml_tensor *Q = ggml_reshape_4d(ctx, q_proj, this->head_dim, this->num_heads, N_q, B);
    ggml_tensor *K = ggml_reshape_4d(ctx, k_proj, this->head_dim, this->num_heads, N_kv, B);
    ggml_tensor *V = ggml_reshape_4d(ctx, v_proj, this->head_dim, this->num_heads, N_kv, B);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    ggml_tensor *attn_out = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f / sqrtf((float)this->head_dim), 0.0f, 0 );

    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    attn_out = ggml_reshape_3d(ctx, attn_out, C, N_q, B);

    ggml_tensor *out = ggml_mul_mat(ctx, this->proj_weight, attn_out);
    if (this->proj_bias) out = ggml_add_inplace(ctx, out, ggml_reshape_3d(ctx, this->proj_bias, C,1,1));
    return out;
}

// --- SanaMSBlock ---
void SanaMSBlock::init_weights(ggml_context* ctx, ggml_type wtype) {
    norm1.init_weights(ctx, wtype, hidden_size);
    attn.d_model = hidden_size; attn.num_heads = num_heads; if (num_heads > 0) attn.head_dim = hidden_size / num_heads; else attn.head_dim = hidden_size;
    attn.init_weights(ctx, wtype);

    cross_attn.d_model = hidden_size; cross_attn.num_heads = num_heads;
    cross_attn.d_cond = hidden_size; // Defaulting, should be set from DITModelParams.text_embed_dim
    if (num_heads > 0) cross_attn.head_dim = hidden_size / num_heads; else cross_attn.head_dim = hidden_size;
    cross_attn.init_weights(ctx, wtype);

    norm2.init_weights(ctx, wtype, hidden_size);
    mlp.C_in = hidden_size; mlp.C_hidden = static_cast<int>(hidden_size * mlp_ratio); mlp.C_out = hidden_size; mlp.kernel_size = 3;
    mlp.init_weights(ctx, wtype);
}
void SanaMSBlock::load_weights(const std::string& prefix, ggml_context *ctx_meta, std::map<std::string, ggml_tensor*>& tensors_map_in) {
    norm1.load_weights(prefix + ".norm1", ctx_meta, tensors_map_in);
    attn.load_weights(prefix + ".attn", ctx_meta, tensors_map_in);
    cross_attn.load_weights(prefix + ".cross_attn", ctx_meta, tensors_map_in);
    norm2.load_weights(prefix + ".norm2", ctx_meta, tensors_map_in);
    mlp.load_weights(prefix + ".mlp", ctx_meta, tensors_map_in);
}
ggml_tensor* SanaMSBlock::forward(ggml_context *ctx, ggml_tensor *x_in, ggml_tensor *y_cond, ggml_tensor *t_mod,
                                   ggml_tensor *cross_mask, int H_feat, int W_feat,
                                   ggml_tensor *image_rotary_emb_cos, ggml_tensor *image_rotary_emb_sin) {
    const int64_t C = x_in->ne[0];
    const int64_t B = x_in->ne[2];

    ggml_tensor *shift_msa = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 0*C*ggml_element_size(t_mod));
    ggml_tensor *scale_msa = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 1*C*ggml_element_size(t_mod));
    ggml_tensor *gate_msa  = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 2*C*ggml_element_size(t_mod));
    ggml_tensor *shift_mlp = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 3*C*ggml_element_size(t_mod));
    ggml_tensor *scale_mlp = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 4*C*ggml_element_size(t_mod));
    ggml_tensor *gate_mlp  = ggml_view_2d(ctx, t_mod, C, B, t_mod->nb[1], 5*C*ggml_element_size(t_mod));

    ggml_tensor *x = x_in;
    ggml_tensor *norm1_out = norm1.forward(ctx, x);
    ggml_tensor *sa_in = sana_t2i_modulate(ctx, norm1_out,
        ggml_reshape_3d(ctx, shift_msa, C,1,B),
        ggml_reshape_3d(ctx, scale_msa, C,1,B)
    );
    ggml_tensor *sa_out = attn.forward(ctx, sa_in, image_rotary_emb_cos, image_rotary_emb_sin);

    ggml_tensor* reshaped_gate_msa = ggml_reshape_3d(ctx, gate_msa, C, 1, B);
    sa_out = ggml_mul(ctx, sa_out, reshaped_gate_msa);

    x = ggml_add(ctx, x, sa_out);

    cross_attn.d_cond = y_cond->ne[0]; // Set d_cond before forward pass if it can vary
    ggml_tensor *cross_attn_out = cross_attn.forward(ctx, x, y_cond, cross_mask);
    x = ggml_add(ctx, x, cross_attn_out);

    ggml_tensor *norm2_out = norm2.forward(ctx, x);
    ggml_tensor *mlp_in = sana_t2i_modulate(ctx, norm2_out,
        ggml_reshape_3d(ctx, shift_mlp, C,1,B),
        ggml_reshape_3d(ctx, scale_mlp, C,1,B)
    );
    mlp.C_in = mlp_in->ne[0];
    mlp.C_out = mlp_in->ne[0];
    mlp.C_hidden = static_cast<int>(mlp_in->ne[0] * this->mlp_ratio);
    mlp.depth_conv_groups = mlp.C_hidden * 2;


    ggml_tensor *mlp_out = mlp.forward(ctx, mlp_in, H_feat, W_feat);

    ggml_tensor* reshaped_gate_mlp = ggml_reshape_3d(ctx, gate_mlp, C, 1, B);
    mlp_out = ggml_mul(ctx, mlp_out, reshaped_gate_mlp);

    x = ggml_add(ctx, x, mlp_out);
    return x;
}

// --- SanaDITModel ---
SanaDITModel::SanaDITModel(const SanaDITModelParams& p) :
    params(p),
    y_norm(p.y_norm_eps),
    final_norm(1e-6f, false)
{
    if (!p.y_norm_active) {
        y_norm.weight = nullptr;
    }
    blocks.reserve(p.depth);
    for (int i = 0; i < p.depth; ++i) {
        blocks.emplace_back(p.hidden_size, p.num_heads, p.text_embed_dim, p.mlp_ratio);
    }
}

bool SanaDITModel::load_params_from_gguf(gguf_context *ctx_gguf) {
    const char* prefix = "sana_dit.";
    std::string key;
    auto get_u32_optional = [&](const char* suffix, int& val, int default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_UINT32) val = gguf_get_val_u32(ctx_gguf, k_idx);
        else if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_INT32) val = gguf_get_val_i32(ctx_gguf, k_idx);
        else val = default_val;
    };
    auto get_f32_optional = [&](const char* suffix, float& val, float default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_FLOAT32) val = gguf_get_val_f32(ctx_gguf, k_idx);
        else val = default_val;
    };
     auto get_bool_optional = [&](const char* suffix, bool& val, bool default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_BOOL) val = gguf_get_val_bool(ctx_gguf, k_idx);
        else val = default_val;
    };

    get_u32_optional("patch_size", params.patch_size, 2);
    get_u32_optional("in_channels_vae", params.in_channels_vae, 4);
    get_u32_optional("hidden_size", params.hidden_size, 1152);
    get_u32_optional("depth", params.depth, 28);
    get_u32_optional("num_heads", params.num_heads, 16);
    get_u32_optional("out_channels_vae", params.out_channels_vae, 4);
    get_f32_optional("mlp_ratio", params.mlp_ratio, 4.0f);
    get_bool_optional("y_norm_active", params.y_norm_active, true);
    get_f32_optional("y_norm_eps", params.y_norm_eps, 1e-5f);
    get_f32_optional("y_norm_scale_factor", params.y_norm_scale_factor, 0.01f);
    get_u32_optional("text_embed_dim", params.text_embed_dim, 2048);
    get_u32_optional("timestep_freq_embed_dim", params.timestep_freq_embed_dim, 256);

    get_bool_optional("is_sprint_model", params.is_sprint_model, false);
    get_f32_optional("sprint_sigma_data", params.sprint_sigma_data, 0.5f);
    get_bool_optional("sprint_cfg_embed", params.sprint_cfg_embed, false);
    get_f32_optional("sprint_cfg_embed_scale", params.sprint_cfg_embed_scale, 1.0f);
    get_f32_optional("sprint_timestep_norm_scale_factor", params.sprint_timestep_norm_scale_factor, 1000.0f);

    if ( (int)blocks.size() != params.depth ||
         ( !blocks.empty() && (blocks[0].hidden_size != params.hidden_size || blocks[0].num_heads != params.num_heads) ) ) {
        blocks.clear();
        blocks.reserve(params.depth);
        for (int i = 0; i < params.depth; ++i) {
            blocks.emplace_back(params.hidden_size, params.num_heads, params.text_embed_dim, params.mlp_ratio);
        }
    }
    if (!params.y_norm_active) y_norm.weight = nullptr;
    else { y_norm.eps = params.y_norm_eps; }

    return true;
}

void SanaDITModel::init_weights(ggml_context *ctx_w, ggml_type wtype) {
    char name_buf[256];
    auto add_to_map = [&](ggml_tensor* t, const std::string& n){ if(t) this->tensors_map[n] = t; };

    snprintf(name_buf, sizeof(name_buf), "sana_dit.x_embedder.conv.weight");
    x_embedder_conv_w = ggml_new_tensor_4d(ctx_w, wtype, params.patch_size, params.patch_size, params.in_channels_vae, params.hidden_size);
    ggml_set_name(x_embedder_conv_w, name_buf); add_to_map(x_embedder_conv_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.x_embedder.conv.bias");
    x_embedder_conv_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.hidden_size);
    ggml_set_name(x_embedder_conv_b, name_buf); add_to_map(x_embedder_conv_b, name_buf);

    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_embedder.mlp.fc1.weight");
    t_embedder_mlp_fc1_w = ggml_new_tensor_2d(ctx_w, wtype, params.timestep_freq_embed_dim, params.hidden_size); ggml_set_name(t_embedder_mlp_fc1_w, name_buf); add_to_map(t_embedder_mlp_fc1_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_embedder.mlp.fc1.bias");
    t_embedder_mlp_fc1_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.hidden_size); ggml_set_name(t_embedder_mlp_fc1_b, name_buf); add_to_map(t_embedder_mlp_fc1_b, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_embedder.mlp.fc2.weight");
    t_embedder_mlp_fc2_w = ggml_new_tensor_2d(ctx_w, wtype, params.hidden_size, params.hidden_size); ggml_set_name(t_embedder_mlp_fc2_w, name_buf); add_to_map(t_embedder_mlp_fc2_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_embedder.mlp.fc2.bias");
    t_embedder_mlp_fc2_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.hidden_size); ggml_set_name(t_embedder_mlp_fc2_b, name_buf); add_to_map(t_embedder_mlp_fc2_b, name_buf);

    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_block.linear.weight");
    t_block_linear_w = ggml_new_tensor_2d(ctx_w, wtype, params.hidden_size, 6 * params.hidden_size); ggml_set_name(t_block_linear_w, name_buf); add_to_map(t_block_linear_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.t_block.linear.bias");
    t_block_linear_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 6 * params.hidden_size); ggml_set_name(t_block_linear_b, name_buf); add_to_map(t_block_linear_b, name_buf);

    if (params.is_sprint_model && params.sprint_cfg_embed) {
        snprintf(name_buf, sizeof(name_buf), "sana_dit.sprint_cfg_embedding.weight");
        sprint_cfg_embedding_w = ggml_new_tensor_2d(ctx_w, wtype, 1, params.hidden_size);
        ggml_set_name(sprint_cfg_embedding_w, name_buf); add_to_map(sprint_cfg_embedding_w, name_buf);
    } else {
        sprint_cfg_embedding_w = nullptr;
    }

    snprintf(name_buf, sizeof(name_buf), "sana_dit.y_proj.fc1.weight");
    y_proj_fc1_w = ggml_new_tensor_2d(ctx_w, wtype, params.text_embed_dim, params.hidden_size); ggml_set_name(y_proj_fc1_w, name_buf); add_to_map(y_proj_fc1_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.y_proj.fc1.bias");
    y_proj_fc1_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.hidden_size); ggml_set_name(y_proj_fc1_b, name_buf); add_to_map(y_proj_fc1_b, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.y_proj.fc2.weight");
    y_proj_fc2_w = ggml_new_tensor_2d(ctx_w, wtype, params.hidden_size, params.hidden_size); ggml_set_name(y_proj_fc2_w, name_buf); add_to_map(y_proj_fc2_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.y_proj.fc2.bias");
    y_proj_fc2_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.hidden_size); ggml_set_name(y_proj_fc2_b, name_buf); add_to_map(y_proj_fc2_b, name_buf);

    if (params.y_norm_active) {
         y_norm.init_weights(ctx_w, wtype, params.hidden_size);
         snprintf(name_buf, sizeof(name_buf), "sana_dit.y_norm.weight"); if(y_norm.weight) {ggml_set_name(y_norm.weight, name_buf); add_to_map(y_norm.weight, name_buf);}
    }

    for (int i = 0; i < params.depth; ++i) {
        snprintf(name_buf, sizeof(name_buf), "sana_dit.blocks.%d", i);
        blocks[i].cross_attn.d_cond = params.text_embed_dim; // Ensure d_cond is set before init_weights
        blocks[i].init_weights(ctx_w, wtype);
    }

    if(final_norm.elementwise_affine) {
        final_norm.init_weights(ctx_w, wtype, params.hidden_size);
        snprintf(name_buf, sizeof(name_buf), "sana_dit.final_norm.weight"); if(final_norm.weight) {ggml_set_name(final_norm.weight, name_buf); add_to_map(final_norm.weight, name_buf);}
        snprintf(name_buf, sizeof(name_buf), "sana_dit.final_norm.bias"); if(final_norm.bias) {ggml_set_name(final_norm.bias, name_buf); add_to_map(final_norm.bias, name_buf);}
    }

    snprintf(name_buf, sizeof(name_buf), "sana_dit.final_linear.weight");
    final_linear_weight = ggml_new_tensor_2d(ctx_w, wtype, params.hidden_size, params.patch_size * params.patch_size * params.out_channels_vae);
    ggml_set_name(final_linear_weight, name_buf); add_to_map(final_linear_weight, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.final_linear.bias");
    final_linear_bias = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, params.patch_size * params.patch_size * params.out_channels_vae);
    ggml_set_name(final_linear_bias, name_buf); add_to_map(final_linear_bias, name_buf);

    snprintf(name_buf, sizeof(name_buf), "sana_dit.final_adaln.linear.weight");
    final_adaLN_modulation_linear_w = ggml_new_tensor_2d(ctx_w, wtype, params.hidden_size , 2*params.hidden_size);
    ggml_set_name(final_adaLN_modulation_linear_w, name_buf); add_to_map(final_adaLN_modulation_linear_w, name_buf);
    snprintf(name_buf, sizeof(name_buf), "sana_dit.final_adaln.linear.bias");
    final_adaLN_modulation_linear_b = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 2*params.hidden_size);
    ggml_set_name(final_adaLN_modulation_linear_b, name_buf); add_to_map(final_adaLN_modulation_linear_b, name_buf);
}

bool SanaDITModel::load_weights_from_gguf(ggml_context *ctx_meta, ggml_backend_buffer_t buffer) {
    (void)buffer;
    char name_buf[256];
    std::string current_prefix_str;

    auto load_tensor_fn = [&](const std::string& tensor_name_suffix, ggml_tensor*& target_tensor, bool required = true) {
        std::string full_name = current_prefix_str + tensor_name_suffix;
        target_tensor = ggml_get_tensor(ctx_meta, full_name.c_str());
        if (target_tensor) {
            this->tensors_map[full_name] = target_tensor;
        } else if (required) {
            fprintf(stderr, "SanaDiTModel: Required tensor %s not found.\n", full_name.c_str());
            return false;
        } else {
            target_tensor = nullptr;
        }
        return true;
    };

    current_prefix_str = "sana_dit.x_embedder.conv.";
    if (!load_tensor_fn("weight", x_embedder_conv_w)) return false;
    load_tensor_fn("bias", x_embedder_conv_b, false);

    current_prefix_str = "sana_dit.t_embedder.mlp.fc1.";
    if (!load_tensor_fn("weight", t_embedder_mlp_fc1_w)) return false;
    if (!load_tensor_fn("bias",   t_embedder_mlp_fc1_b)) return false;
    current_prefix_str = "sana_dit.t_embedder.mlp.fc2.";
    if (!load_tensor_fn("weight", t_embedder_mlp_fc2_w)) return false;
    if (!load_tensor_fn("bias",   t_embedder_mlp_fc2_b)) return false;
    current_prefix_str = "sana_dit.t_block.linear.";
    if (!load_tensor_fn("weight",     t_block_linear_w)) return false;
    if (!load_tensor_fn("bias",       t_block_linear_b)) return false;

    if (params.is_sprint_model && params.sprint_cfg_embed) {
        current_prefix_str = "sana_dit.sprint_cfg_embedding.";
        load_tensor_fn("weight", sprint_cfg_embedding_w, false);
    }

    current_prefix_str = "sana_dit.y_proj.fc1.";
    if (!load_tensor_fn("weight", y_proj_fc1_w)) return false;
    if (!load_tensor_fn("bias",   y_proj_fc1_b)) return false;
    current_prefix_str = "sana_dit.y_proj.fc2.";
    if (!load_tensor_fn("weight", y_proj_fc2_w)) return false;
    if (!load_tensor_fn("bias",   y_proj_fc2_b)) return false;

    if (params.y_norm_active) {
        y_norm.load_weights("sana_dit.y_norm", ctx_meta, this->tensors_map);
        if (!y_norm.weight && params.y_norm_active) {
            // This can be an issue if y_norm.weight is essential for y_norm_active=true
            // For now, it's just a warning if not found by get_tensor_from_map_optional
            // fprintf(stderr, "SanaDiTModel: y_norm.weight not found after load_weights but y_norm_active is true.\n");
        }
    }

    for(int i=0; i<params.depth; ++i) {
        snprintf(name_buf, sizeof(name_buf), "sana_dit.blocks.%d.", i);
        blocks[i].load_weights(name_buf, ctx_meta, this->tensors_map);
    }

    if (final_norm.elementwise_affine) {
         final_norm.load_weights("sana_dit.final_norm.", ctx_meta, this->tensors_map);
         if (!final_norm.weight && final_norm.elementwise_affine) { // If it's affine, weight should exist
            fprintf(stderr, "SanaDiTModel: final_norm.weight not found but final_norm is affine.\n");
            return false;
         }
    }
    current_prefix_str = "sana_dit.final_linear.";
    if (!load_tensor_fn("weight", final_linear_weight)) return false;
    if (!load_tensor_fn("bias", final_linear_bias)) return false;
    current_prefix_str = "sana_dit.final_adaln.linear.";
    if (!load_tensor_fn("weight", final_adaLN_modulation_linear_w)) return false;
    if (!load_tensor_fn("bias", final_adaLN_modulation_linear_b)) return false;

    return true;
}


ggml_cgraph* SanaDITModel::build_graph(
    ggml_context *ctx,
    ggml_tensor *x_latent_input,
    ggml_tensor *raw_timestep_input,
    ggml_tensor *raw_y_embed,
    ggml_tensor *text_mask,
    ggml_tensor *cfg_scale_tensor
) {
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, SANA_DIT_GRAPH_SIZE, false);

    const int64_t B_orig = x_latent_input->ne[0];
    const int64_t H_l = x_latent_input->ne[2];
    const int64_t W_l = x_latent_input->ne[3];

    ggml_tensor* x = ggml_cont(ctx, ggml_permute(ctx, x_latent_input, 3, 2, 1, 0));
    x = ggml_conv_2d(ctx, x_embedder_conv_w, x, params.patch_size, params.patch_size, 0, 0, 1, 1);
    if (x_embedder_conv_b) {
        x = ggml_add_inplace(ctx, x, ggml_reshape_4d(ctx, x_embedder_conv_b, 1, 1, params.hidden_size, 1));
    }
    int H_feat = H_l / params.patch_size;
    int W_feat = W_l / params.patch_size;
    int N_patches = H_feat * W_feat;
    x = ggml_permute(ctx, x, 2, 0, 1, 3);
    x = ggml_reshape_3d(ctx, x, params.hidden_size, N_patches, B_orig);
    ggml_set_name(x, "patched_input");

    ggml_tensor* t_processed_for_blocks;
    ggml_tensor* t_for_final_layer_mod;

    if (params.is_sprint_model) {
        ggml_tensor* s_scm_input = raw_timestep_input;
        if (ggml_is_scalar(s_scm_input) || (s_scm_input->ne[0] == 1 && B_orig > 1 && s_scm_input->ne[1] != B_orig ) ) {
            s_scm_input = ggml_reshape_1d(ctx, s_scm_input, 1);
            s_scm_input = ggml_repeat(ctx, s_scm_input, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B_orig) );
        }
        s_scm_input = ggml_reshape_2d(ctx, s_scm_input, 1, B_orig);

        ggml_tensor* sin_s = ggml_sin(ctx, s_scm_input);
        ggml_tensor* cos_s = ggml_cos(ctx, s_scm_input);
        ggml_tensor* sin_s_plus_cos_s = ggml_add(ctx, sin_s, cos_s);
        ggml_tensor* t_scm = ggml_div(ctx, sin_s, sin_s_plus_cos_s);
        ggml_set_name(t_scm, "t_scm_transformed");

        ggml_tensor* x_scaled_for_sprint = ggml_scale(ctx, x, 1.0f / params.sprint_sigma_data);
        ggml_tensor* t_sq = ggml_mul(ctx, t_scm, t_scm);

        ggml_tensor* one_val_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ((float*)one_val_tensor->data)[0] = 1.0f;
        one_val_tensor = ggml_reshape_2d(ctx, one_val_tensor, 1, 1); // Make it (1,1) to broadcast with (1,B) t_scm

        ggml_tensor* one_minus_t = ggml_sub(ctx, ggml_repeat(ctx, one_val_tensor, t_scm), t_scm);
        ggml_tensor* one_minus_t_sq = ggml_mul(ctx, one_minus_t, one_minus_t);
        ggml_tensor* scale_factor_scm_input_sqrt_arg = ggml_add(ctx, t_sq, one_minus_t_sq);
        ggml_tensor* scale_factor_scm_input = ggml_sqrt(ctx, scale_factor_scm_input_sqrt_arg);
        x = ggml_mul(ctx, x_scaled_for_sprint, ggml_repeat(ctx, scale_factor_scm_input, x_scaled_for_sprint));
        ggml_set_name(x, "scm_transformed_input_latent");

        ggml_tensor* pretrain_timestep_val = ggml_scale(ctx, t_scm, params.sprint_timestep_norm_scale_factor);

        // Placeholder for sinusoidal expansion of pretrain_timestep_val before MLP
        // This requires pretrain_timestep_val (1,B) to become (FreqEmbSize, B)
        ggml_tensor* pretrain_timestep_freq_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.timestep_freq_embed_dim, B_orig); // Placeholder
        // --- Actual Sinusoidal Embedding would go here ---

        t_processed_for_blocks = ggml_mul_mat(ctx, t_embedder_mlp_fc1_w, pretrain_timestep_freq_emb); // Needs correct input
        t_processed_for_blocks = ggml_add_inplace(ctx, t_processed_for_blocks, ggml_reshape_2d(ctx, t_embedder_mlp_fc1_b, params.hidden_size, 1));
        t_processed_for_blocks = ggml_silu_inplace(ctx, t_processed_for_blocks);
        t_processed_for_blocks = ggml_mul_mat(ctx, t_embedder_mlp_fc2_w, t_processed_for_blocks);
        t_processed_for_blocks = ggml_add_inplace(ctx, t_processed_for_blocks, ggml_reshape_2d(ctx, t_embedder_mlp_fc2_b, params.hidden_size, 1));
        ggml_set_name(t_processed_for_blocks, "scm_processed_timestep_for_blocks");

        if (params.sprint_cfg_embed && cfg_scale_tensor) {
            ggml_tensor* cfg_val = cfg_scale_tensor;
            if (ggml_is_scalar(cfg_val) || (cfg_val->ne[0] == 1 && B_orig > 1 && cfg_val->ne[1] != B_orig) ) {
                 cfg_val = ggml_reshape_1d(ctx, cfg_val, 1);
                 cfg_val = ggml_repeat(ctx, cfg_val, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B_orig));
            }
            cfg_val = ggml_reshape_2d(ctx, cfg_val, 1, B_orig);

            ggml_tensor* cfg_embedded;
            if (sprint_cfg_embedding_w) {
                cfg_embedded = ggml_mul_mat(ctx, sprint_cfg_embedding_w, cfg_val);
            } else {
                cfg_embedded = ggml_scale(ctx, cfg_val, params.sprint_cfg_embed_scale);
                cfg_embedded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, params.hidden_size, B_orig); // Placeholder
            }
            t_processed_for_blocks = ggml_add(ctx, t_processed_for_blocks, cfg_embedded);
            ggml_set_name(t_processed_for_blocks, "scm_timestep_with_cfg");
        }
        t_for_final_layer_mod = t_processed_for_blocks;

    } else {
        ggml_tensor* t_emb_permuted = ggml_cont(ctx, ggml_permute(ctx, raw_timestep_input, 1, 0, 2, 3));
        t_processed_for_blocks = ggml_mul_mat(ctx, t_embedder_mlp_fc1_w, t_emb_permuted);
        t_processed_for_blocks = ggml_add_inplace(ctx, t_processed_for_blocks, ggml_reshape_2d(ctx, t_embedder_mlp_fc1_b, params.hidden_size, 1));
        t_processed_for_blocks = ggml_silu_inplace(ctx, t_processed_for_blocks);
        t_processed_for_blocks = ggml_mul_mat(ctx, t_embedder_mlp_fc2_w, t_processed_for_blocks);
        t_processed_for_blocks = ggml_add_inplace(ctx, t_processed_for_blocks, ggml_reshape_2d(ctx, t_embedder_mlp_fc2_b, params.hidden_size, 1));
        t_for_final_layer_mod = t_processed_for_blocks;
    }
    ggml_set_name(t_for_final_layer_mod, "final_processed_timestep_embed");

    ggml_tensor* t_mod_input_for_blocks = ggml_silu(ctx, t_processed_for_blocks);
    ggml_tensor* t_mod_for_blocks_out = ggml_mul_mat(ctx, t_block_linear_w, t_mod_input_for_blocks); // Corrected variable name
    t_mod_for_blocks_out = ggml_add_inplace(ctx, t_mod_for_blocks_out, ggml_reshape_2d(ctx, t_block_linear_b, 6*params.hidden_size, 1));
    ggml_set_name(t_mod_for_blocks_out, "timestep_modulation_for_blocks");

    ggml_tensor* y_embed_permuted = ggml_cont(ctx, ggml_permute(ctx, raw_y_embed, 2, 1, 0, 3));
    ggml_tensor* y_proj = ggml_mul_mat(ctx, y_proj_fc1_w, y_embed_permuted);
    y_proj = ggml_add_inplace(ctx, y_proj, ggml_reshape_3d(ctx, y_proj_fc1_b, params.hidden_size, 1, 1));
    y_proj = ggml_gelu_inplace(ctx, y_proj);
    y_proj = ggml_mul_mat(ctx, y_proj_fc2_w, y_proj);
    y_proj = ggml_add_inplace(ctx, y_proj, ggml_reshape_3d(ctx, y_proj_fc2_b, params.hidden_size, 1, 1));
    ggml_set_name(y_proj, "projected_text_embed_raw");

    if (params.y_norm_active && y_norm.weight) {
        y_proj = y_norm.forward(ctx, y_proj);
        y_proj = ggml_scale(ctx, y_proj, params.y_norm_scale_factor);
        ggml_set_name(y_proj, "normed_projected_text_embed");
    }

    ggml_tensor* rope_cos = nullptr;
    ggml_tensor* rope_sin = nullptr;
    ggml_tensor* current_cross_mask = text_mask;

    ggml_tensor* x_input_to_blocks_loop = x;

    for (size_t i = 0; i < blocks.size(); ++i) {
        char name_buf[64];
        snprintf(name_buf, sizeof(name_buf), "block_%zu_input", i);
        ggml_set_name(x, name_buf);
        x = blocks[i].forward(ctx, x, y_proj, t_mod_for_blocks_out, current_cross_mask, H_feat, W_feat, rope_cos, rope_sin); // Use t_mod_for_blocks_out
        snprintf(name_buf, sizeof(name_buf), "block_%zu_out", i);
        ggml_set_name(x, name_buf);
    }

    ggml_tensor* dit_output_after_blocks = x;

    ggml_tensor* final_mod_input_silu = ggml_silu(ctx, t_for_final_layer_mod);
    ggml_tensor* final_mod_params = ggml_mul_mat(ctx, final_adaLN_modulation_linear_w, final_mod_input_silu);
    final_mod_params = ggml_add_inplace(ctx, final_mod_params, ggml_reshape_2d(ctx, final_adaLN_modulation_linear_b, 2*params.hidden_size, 1));

    ggml_tensor* shift_final = ggml_view_2d(ctx, final_mod_params, params.hidden_size, B_orig, final_mod_params->nb[1], 0);
    ggml_tensor* scale_final = ggml_view_2d(ctx, final_mod_params, params.hidden_size, B_orig, final_mod_params->nb[1], params.hidden_size * ggml_element_size(final_mod_params));

    x = final_norm.forward(ctx, dit_output_after_blocks);
    x = sana_modulate(ctx, x, shift_final, scale_final);

    x = ggml_mul_mat(ctx, final_linear_weight, x);
    if (final_linear_bias) {
         x = ggml_add_inplace(ctx, x, ggml_reshape_3d(ctx, final_linear_bias, final_linear_bias->ne[0],1,1));
    }
    ggml_set_name(x, "output_before_sprint_final_transform");

    if (params.is_sprint_model) {
        ggml_tensor* t_scm = ggml_get_tensor(ctx, "t_scm_transformed");
        if (!t_scm) {
            t_scm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); // Create a scalar tensor
            ((float*)t_scm->data)[0] = 0.0f; // Set its value
            t_scm = ggml_reshape_2d(ctx, t_scm, 1, 1); // Reshape to (1,1) for broadcasting with B if needed
            if (B_orig > 1) t_scm = ggml_repeat(ctx, t_scm, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B_orig)); // Repeat to (1,B)
        }


        ggml_tensor* one_val_sprint = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ((float*)one_val_sprint->data)[0] = 1.0f;
        one_val_sprint = ggml_reshape_2d(ctx, one_val_sprint, 1,1);
        ggml_tensor* two_val_sprint = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ((float*)two_val_sprint->data)[0] = 2.0f;
        two_val_sprint = ggml_reshape_2d(ctx, two_val_sprint, 1,1);


        ggml_tensor* one_minus_2t = ggml_sub(ctx, ggml_repeat(ctx,one_val_sprint, t_scm), ggml_mul(ctx, ggml_repeat(ctx,two_val_sprint, t_scm), t_scm));

        ggml_tensor* term1_num = ggml_mul(ctx, ggml_repeat(ctx, one_minus_2t, x_input_to_blocks_loop), x_input_to_blocks_loop);

        ggml_tensor* t_sq = ggml_mul(ctx, t_scm, t_scm);
        ggml_tensor* term2_factor = ggml_add(ctx, one_minus_2t, ggml_mul(ctx, ggml_repeat(ctx,two_val_sprint, t_sq), t_sq));

        ggml_tensor* current_dit_output = x; // This is the output from final_linear layer
        ggml_tensor* term2_num = ggml_mul(ctx, ggml_repeat(ctx, term2_factor, current_dit_output), current_dit_output);

        ggml_tensor* numerator = ggml_add(ctx, term1_num, term2_num);

        ggml_tensor* one_minus_t = ggml_sub(ctx, ggml_repeat(ctx,one_val_sprint,t_scm), t_scm);
        ggml_tensor* one_minus_t_sq = ggml_mul(ctx, one_minus_t, one_minus_t);
        ggml_tensor* denominator_sqrt_arg = ggml_add(ctx, t_sq, one_minus_t_sq);
        ggml_tensor* denominator = ggml_sqrt(ctx, denominator_sqrt_arg);

        x = ggml_div(ctx, numerator, ggml_repeat(ctx, denominator, numerator));
        ggml_set_name(x, "sprint_final_output_unpatched");
    } else {
        ggml_set_name(x, "final_projection_before_unpatch");
    }

    ggml_build_forward_expand(gf, x);
    return gf;
}
