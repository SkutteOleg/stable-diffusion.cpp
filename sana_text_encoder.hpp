#ifndef SANA_TEXT_ENCODER_HPP
#define SANA_TEXT_ENCODER_HPP

#include "ggml.h"
#include "ggml-backend.h" // Included for ggml_backend_buffer_t
#include <vector>
#include <string>
#include <map>
#include <array>

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct gguf_context;

struct SanaVocab {
    std::map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;
    std::vector<float> id_to_score;

    int32_t bos_token_id = 0;
    int32_t eos_token_id = 1;
    int32_t pad_token_id = -1;
    int32_t unk_token_id = 2;

    enum class Type {
        SPM,
        BPE
    } type = Type::SPM;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::vector<std::string> bpe_merges;

    bool load_from_gguf(gguf_context *ctx_gguf); // Reads metadata from GGUF
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos, bool add_eos) const;
  private:
    std::vector<int32_t> tokenize_spm(const std::string& text) const;
    std::vector<int32_t> tokenize_bpe(const std::string& text) const;
};

struct SanaTextEncoderParams {
    uint32_t n_vocab = 32000;
    uint32_t n_embd = 768;
    uint32_t n_layer = 12;
    uint32_t n_head = 12;
    uint32_t n_ff = 3072;
    float norm_eps = 1e-6f;
    uint32_t n_rot = 64;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    uint32_t n_ctx_train = 512;

    uint32_t n_rel_attn_bkts = 0;

    bool loaded = false;
};

struct SanaTextEncoderLayer {
    // Attention
    ggml_tensor *attn_q_w;
    ggml_tensor *attn_k_w;
    ggml_tensor *attn_v_w;
    ggml_tensor *attn_o_w;
    ggml_tensor *attn_q_b;
    ggml_tensor *attn_k_b;
    ggml_tensor *attn_v_b;
    ggml_tensor *attn_o_b;

    // Normalization
    ggml_tensor *attn_norm_w;
    ggml_tensor *attn_norm_b;

    // FFN
    ggml_tensor *ffn_gate_w;
    ggml_tensor *ffn_down_w;
    ggml_tensor *ffn_up_w;
    ggml_tensor *ffn_gate_b;
    ggml_tensor *ffn_down_b;
    ggml_tensor *ffn_up_b;

    // Normalization
    ggml_tensor *ffn_norm_w;
    ggml_tensor *ffn_norm_b;
};

struct SanaTextEncoderModel {
    SanaTextEncoderParams params;
    SanaVocab vocab;

    ggml_tensor *tok_embeddings_weight;
    ggml_tensor *pos_embeddings_weight;

    std::vector<SanaTextEncoderLayer> layers;

    ggml_tensor *final_norm_w;
    ggml_tensor *final_norm_b;

    std::map<std::string, ggml_tensor *> tensors_map;

    SanaTextEncoderModel() = default;

    bool load_params_from_gguf(gguf_context *ctx_gguf);
    // ctx_weights is the ggml_context where GGUF has loaded tensor *metadata* (no_alloc=true)
    // buffer is where actual tensor data might be allocated by the backend if not CPU.
    bool load_weights_from_gguf(ggml_context *ctx_weights_from_gguf, ggml_backend_buffer_t buffer);

    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = false, bool add_eos = false) const {
        return vocab.tokenize(text, add_bos, add_eos);
    }

    ggml_cgraph* build_graph(
        ggml_context *ctx_compute,
        ggml_tensor *token_ids,
        ggml_tensor *input_positions
    );

    void init_weights(ggml_context *ctx_weights, ggml_type wtype);
};

ggml_tensor* get_tensor_from_map(const std::map<std::string, ggml_tensor*>& tensors_map, const std::string& name);

#define SANA_TEXT_ENCODER_GRAPH_SIZE 4096

#endif // SANA_TEXT_ENCODER_HPP
