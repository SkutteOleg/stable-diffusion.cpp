#include "sana_text_encoder.hpp"
#include "gguf.h"
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
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

// --- SanaVocab ---

bool SanaVocab::load_from_gguf(gguf_context *ctx_gguf) {
    int tokens_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (tokens_idx == -1) {
        fprintf(stderr, "SanaVocab: Key 'tokenizer.ggml.tokens' not found in GGUF.\n");
        return false;
    }
    int model_type_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.model");
    if (model_type_idx != -1) {
        std::string model_name = std::string(gguf_get_val_str(ctx_gguf, model_type_idx));
        if (model_name == "gpt2" || model_name == "gpt-2" || model_name == "bpe") {
            type = Type::BPE;
            int merges_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
            if (merges_idx != -1) {
                uint32_t n_merges = gguf_get_arr_n(ctx_gguf, merges_idx);
                bpe_merges.resize(n_merges);
                for (uint32_t i = 0; i < n_merges; ++i) {
                    const char* merge_str_c = gguf_get_arr_str(ctx_gguf, merges_idx, i);
                    if (merge_str_c) bpe_merges[i] = std::string(merge_str_c);

                    size_t space_pos = bpe_merges[i].find(' ');
                    if (space_pos != std::string::npos && space_pos > 0 && space_pos < bpe_merges[i].length() - 1) {
                        std::string p1 = bpe_merges[i].substr(0, space_pos);
                        std::string p2 = bpe_merges[i].substr(space_pos + 1);
                        bpe_ranks[std::make_pair(p1,p2)] = i;
                    }
                }
            } else {
                 fprintf(stderr, "SanaVocab: BPE model type specified but no merges found.\n");
            }
        } else {
            type = Type::SPM;
        }
    }

    uint32_t n_vocab_gguf = gguf_get_arr_n(ctx_gguf, tokens_idx);
    id_to_token.resize(n_vocab_gguf);

    int scores_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.scores");
    if (scores_idx != -1 && type == Type::SPM) {
        if (gguf_get_arr_n(ctx_gguf, scores_idx) == n_vocab_gguf) {
             id_to_score.resize(n_vocab_gguf);
        } else {
            fprintf(stderr, "SanaVocab: Warning - scores array size mismatch for SPM.\n");
            scores_idx = -1;
        }
    } else if (type == Type::SPM && scores_idx == -1 && n_vocab_gguf > 0) {
        fprintf(stderr, "SanaVocab: Warning - SPM model type but no scores found (or size mismatch).\n");
    }

    for (uint32_t i = 0; i < n_vocab_gguf; ++i) {
        const char* token_text_c = gguf_get_arr_str(ctx_gguf, tokens_idx, i);
        if (token_text_c == nullptr) {
            fprintf(stderr, "SanaVocab: Failed to get token string for id %u\n", i);
            return false;
        }
        id_to_token[i] = token_text_c;
        token_to_id[id_to_token[i]] = i;
        if (scores_idx != -1 && i < id_to_score.size()) {
            const void* score_data = gguf_get_arr_data(ctx_gguf, scores_idx);
            if (score_data) {
                id_to_score[i] = ((const float *)score_data)[i];
            } else {
                 fprintf(stderr, "SanaVocab: Failed to get score data for id %u (data pointer is null).\n", i);
            }
        }
    }

    auto load_special_token = [&](const char* key_name, int32_t& token_id_member, int32_t default_val_if_not_found) {
        int key_idx = gguf_find_key(ctx_gguf, key_name);
        if (key_idx != -1) {
            if (gguf_get_kv_type(ctx_gguf, key_idx) == GGUF_TYPE_UINT32) {
                 token_id_member = gguf_get_val_u32(ctx_gguf, key_idx);
            } else if (gguf_get_kv_type(ctx_gguf, key_idx) == GGUF_TYPE_INT32) {
                 token_id_member = gguf_get_val_i32(ctx_gguf, key_idx);
            } else {
                token_id_member = default_val_if_not_found;
            }
        } else {
            token_id_member = default_val_if_not_found;
        }
    };

    load_special_token("tokenizer.ggml.bos_token_id", bos_token_id, -1);
    load_special_token("tokenizer.ggml.eos_token_id", eos_token_id, -1);
    load_special_token("tokenizer.ggml.unk_token_id", unk_token_id, -1);

    int pad_id_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.padding_token_id");
    if (pad_id_idx == -1) pad_id_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.pad_token_id");
    if (pad_id_idx != -1) {
        if (gguf_get_kv_type(ctx_gguf, pad_id_idx) == GGUF_TYPE_UINT32) {
            pad_token_id = gguf_get_val_u32(ctx_gguf, pad_id_idx);
        } else if (gguf_get_kv_type(ctx_gguf, pad_id_idx) == GGUF_TYPE_INT32) {
            pad_token_id = gguf_get_val_i32(ctx_gguf, pad_id_idx);
        }
    } else {
        pad_token_id = -1;
    }
    return true;
}

struct SanaSpmSymbol {
    using index = int;
    index prev = -1;
    index next = -1;
    const char *text_ptr = nullptr;
    size_t n = 0;
};

struct SanaSpmBigram {
    struct comparator {
        bool operator()(const SanaSpmBigram& l, const SanaSpmBigram& r) const {
            if (l.score != r.score) {
                return l.score < r.score;
            }
            return l.left > r.left;
        }
    };
    using queue_storage = std::vector<SanaSpmBigram>;
    using queue = std::priority_queue<SanaSpmBigram, queue_storage, comparator>;

    SanaSpmSymbol::index left;
    SanaSpmSymbol::index right;
    float score;
    size_t size;
};

std::vector<int32_t> SanaVocab::tokenize_spm(const std::string& text) const {
    std::vector<int32_t> output_tokens;
    if (text.empty() || id_to_score.empty()) {
        if (!text.empty() && unk_token_id != -1) output_tokens.push_back(unk_token_id);
        return output_tokens;
    }

    std::vector<SanaSpmSymbol> symbols;
    symbols.reserve(text.length());

    int current_symbol_idx = 0;
    size_t current_offs = 0;
    while (current_offs < text.length()) {
        SanaSpmSymbol sym;
        size_t char_len = utf8_len_char(text.c_str() + current_offs, text.length() - current_offs);

        if (char_len == 0) {
            current_offs++;
            continue;
        }
        sym.text_ptr = text.c_str() + current_offs;
        sym.n = char_len;
        sym.prev = current_symbol_idx - 1;
        symbols.push_back(sym);
        current_symbol_idx++;
        current_offs += char_len;
    }

    if (symbols.empty()) return output_tokens;

    for(size_t i = 0; i < symbols.size(); ++i) {
        if (i < symbols.size() - 1) {
            symbols[i].next = i + 1;
        } else {
            symbols[i].next = -1;
        }
    }

    SanaSpmBigram::queue work_queue;
    auto try_add_bigram_spm = [&](SanaSpmSymbol::index left_idx, SanaSpmSymbol::index right_idx) {
        if (left_idx == -1 || right_idx == -1) return;
        if ((size_t)left_idx >= symbols.size() || (size_t)right_idx >= symbols.size()) return;

        const SanaSpmSymbol& left_s = symbols[left_idx];
        const SanaSpmSymbol& right_s = symbols[right_idx];
        if (left_s.n == 0 || right_s.n == 0) return;

        std::string piece(left_s.text_ptr, left_s.n + right_s.n);
        auto it = token_to_id.find(piece);

        if (it != token_to_id.end() && (uint32_t)it->second < id_to_score.size()) {
            work_queue.push({left_idx, right_idx, id_to_score[it->second], piece.length()});
        }
    };

    for (size_t i = 0; i < symbols.size() - 1; ++i) {
        try_add_bigram_spm(i, i + 1);
    }

    while (!work_queue.empty()) {
        SanaSpmBigram bigram = work_queue.top();
        work_queue.pop();

        if ((size_t)bigram.left >= symbols.size() || (size_t)bigram.right >= symbols.size()) continue;

        SanaSpmSymbol& left_sym = symbols[bigram.left];
        SanaSpmSymbol& right_sym = symbols[bigram.right];

        if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
            continue;
        }

        left_sym.n += right_sym.n;
        right_sym.n = 0;

        left_sym.next = right_sym.next;
        if (right_sym.next != -1 && (size_t)right_sym.next < symbols.size()) {
            symbols[right_sym.next].prev = bigram.left;
        }

        try_add_bigram_spm(left_sym.prev, bigram.left);
        try_add_bigram_spm(bigram.left, left_sym.next);
    }

    for (int i = 0; i != -1; i = symbols[i].next) {
        if (symbols[i].n > 0) {
            std::string final_piece(symbols[i].text_ptr, symbols[i].n);
            auto it = token_to_id.find(final_piece);
            if (it != token_to_id.end()) {
                output_tokens.push_back(it->second);
            } else {
                for (size_t j = 0; j < final_piece.length(); ) {
                     size_t char_len = utf8_len_char(final_piece.c_str() + j, final_piece.length() - j);
                     if (char_len == 0) { if (unk_token_id !=-1) output_tokens.push_back(unk_token_id); j++; continue;}
                     std::string char_s = final_piece.substr(j, char_len);
                     auto it_char = token_to_id.find(char_s);
                     if(it_char != token_to_id.end()){
                         output_tokens.push_back(it_char->second);
                     } else {
                        if (unk_token_id != -1) output_tokens.push_back(unk_token_id);
                     }
                     j += char_len;
                }
            }
        }
    }
    return output_tokens;
}

std::vector<int32_t> SanaVocab::tokenize(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int32_t> output;
    if (add_bos && bos_token_id != -1 && (uint32_t)bos_token_id < id_to_token.size()) {
        output.push_back(bos_token_id);
    }

    std::vector<int32_t> tokens;
    if (type == Type::SPM) {
        tokens = tokenize_spm(text);
    } else {
        tokens = tokenize_bpe(text);
    }
    output.insert(output.end(), tokens.begin(), tokens.end());

    if (add_eos && eos_token_id != -1 && (uint32_t)eos_token_id < id_to_token.size()) {
        output.push_back(eos_token_id);
    }
    return output;
}

std::vector<std::string> bpe_pre_tokenize_gpt2(const std::string& text) {
    if (text.empty()) return {};
    std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[0-9]+|[^\s[:alnum:]]+|\s+)");

    std::sregex_token_iterator it(text.begin(), text.end(), re, 0);
    std::sregex_token_iterator end;
    std::vector<std::string> result;

    size_t last_pos = 0;
    for (; it != end; ++it) {
        std::ssub_match sub_match = *it; // Corrected: use ssub_match
        if (sub_match.first != text.begin() + last_pos) { // Check if sub_match.position() > last_pos implicitly
            // This logic for capturing non-matched parts needs to be careful with iterators
            // For now, we rely on the regex to capture all relevant parts.
            // A more robust solution might iterate char-by-char over gaps.
        }
        if (sub_match.length() > 0) {
            result.push_back(sub_match.str());
        }
        last_pos += sub_match.length();
    }
    // If the regex doesn't match the whole string or leaves trailing characters:
    if (last_pos < text.length()) {
        std::string remaining_part = text.substr(last_pos);
         for (size_t k=0; k < remaining_part.length(); ) {
            size_t char_len = utf8_len_char(remaining_part.c_str() + k, remaining_part.length() - k);
            if (char_len == 0) {k++; continue;}
            result.push_back(remaining_part.substr(k, char_len));
            k += char_len;
        }
    }
    // If regex failed entirely and text is not empty
    if (result.empty() && !text.empty()){
         for (size_t k=0; k < text.length(); ) {
            size_t char_len = utf8_len_char(text.c_str() + k, text.length() - k);
            if (char_len == 0) {k++; continue;}
            result.push_back(text.substr(k, char_len));
            k += char_len;
        }
    }
    return result;
}

std::vector<int32_t> SanaVocab::tokenize_bpe(const std::string& text) const {
    std::vector<int32_t> output_tokens;
    if (text.empty()) return output_tokens;

    if (bpe_ranks.empty() && id_to_token.size() < 256) {
        for (size_t k=0; k < text.length(); ) {
            size_t char_len = utf8_len_char(text.c_str() + k, text.length() - k);
            if (char_len == 0) { if (unk_token_id !=-1) output_tokens.push_back(unk_token_id); k++; continue;}
            std::string char_s = text.substr(k, char_len);
            auto it_char = token_to_id.find(char_s);
            if(it_char != token_to_id.end()){ output_tokens.push_back(it_char->second); }
            else { if (unk_token_id != -1) output_tokens.push_back(unk_token_id); }
            k += char_len;
        }
        return output_tokens;
    }

    std::vector<std::string> words = bpe_pre_tokenize_gpt2(text);

    for (const std::string& word : words) {
        if (word.empty()) continue;

        auto word_token_it = token_to_id.find(word);
        if (word_token_it != token_to_id.end()) {
            output_tokens.push_back(word_token_it->second);
            continue;
        }

        std::vector<std::string> sub_tokens;
        for (size_t k=0; k < word.length(); ) {
            size_t char_len = utf8_len_char(word.c_str() + k, word.length() - k);
            if (char_len == 0) {k++; continue;}
            sub_tokens.push_back(word.substr(k, char_len));
            k += char_len;
        }

        if (sub_tokens.empty()) continue;

        bool merged_in_iteration = true;
        while(merged_in_iteration && sub_tokens.size() > 1) {
            merged_in_iteration = false;
            int best_rank = -1;
            int merge_idx = -1;

            for (size_t j = 0; j < sub_tokens.size() - 1; ++j) {
                auto pair = std::make_pair(sub_tokens[j], sub_tokens[j+1]);
                auto rank_it = bpe_ranks.find(pair);
                if (rank_it != bpe_ranks.end()) {
                    if (best_rank == -1 || rank_it->second < best_rank) {
                        best_rank = rank_it->second;
                        merge_idx = j;
                    }
                }
            }

            if (best_rank != -1) {
                std::string merged_token_str = sub_tokens[merge_idx] + sub_tokens[merge_idx+1];
                sub_tokens.erase(sub_tokens.begin() + merge_idx, sub_tokens.begin() + merge_idx + 2);
                sub_tokens.insert(sub_tokens.begin() + merge_idx, merged_token_str);
                merged_in_iteration = true;
            }
        }
        for(const auto& st : sub_tokens) {
            auto it = token_to_id.find(st);
            if (it != token_to_id.end()) output_tokens.push_back(it->second);
            else {
                 for (size_t k=0; k < st.length(); ) {
                    size_t char_len = utf8_len_char(st.c_str() + k, st.length() - k);
                     if (char_len == 0) { if (unk_token_id !=-1) output_tokens.push_back(unk_token_id); k++; continue;}
                    std::string char_s = st.substr(k, char_len);
                    auto it_char = token_to_id.find(char_s);
                     if(it_char != token_to_id.end()){
                         output_tokens.push_back(it_char->second);
                     } else {
                        if (unk_token_id != -1) output_tokens.push_back(unk_token_id);
                     }
                     k += char_len;
                }
            }
        }
    }
    return output_tokens;
}

// --- SanaTextEncoderModel --- (Implementations for load_params, load_weights, init_weights, build_graph remain the same as previous correct version)
bool SanaTextEncoderModel::load_params_from_gguf(gguf_context *ctx_gguf) {
    const char* prefix = "text_encoder.";
    std::string key;

    auto get_u32_optional = [&](const char* suffix, uint32_t& val, uint32_t default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_UINT32) val = gguf_get_val_u32(ctx_gguf, k_idx);
        else val = default_val;
    };
     auto get_f32_optional = [&](const char* suffix, float& val, float default_val) {
        key = std::string(prefix) + suffix;
        int k_idx = gguf_find_key(ctx_gguf, key.c_str());
        if (k_idx != -1 && gguf_get_kv_type(ctx_gguf, k_idx) == GGUF_TYPE_FLOAT32) val = gguf_get_val_f32(ctx_gguf, k_idx);
        else val = default_val;
    };

    if (!vocab.load_from_gguf(ctx_gguf)) {
        fprintf(stderr, "Failed to load vocab for SanaTextEncoderModel during param loading.\n");
        return false;
    }
    params.n_vocab = vocab.id_to_token.empty() ? 32000 : vocab.id_to_token.size();

    get_u32_optional("embedding_length", params.n_embd, 768);
    get_u32_optional("block_count", params.n_layer, 12);
    get_u32_optional("attention.head_count", params.n_head, 12);
    get_u32_optional("feed_forward_length", params.n_ff, params.n_embd * 4);

    get_f32_optional("attention.layer_norm_rms_epsilon", params.norm_eps, -1.0f);
    if (params.norm_eps == -1.0f) {
         get_f32_optional("attention.layer_norm_epsilon", params.norm_eps, 1e-6f);
    }

    uint32_t default_rot = (params.n_head > 0 && params.n_embd > 0) ? (params.n_embd / params.n_head) : 64;
    get_u32_optional("rope.dimension_count", params.n_rot, default_rot);

    get_f32_optional("rope.freq_base", params.rope_freq_base, 10000.0f);
    get_f32_optional("rope.scaling_factor", params.rope_freq_scale, 1.0f);
    get_u32_optional("context_length", params.n_ctx_train, 512);
    get_u32_optional("attention.relative_attention_num_buckets", params.n_rel_attn_bkts, 0);

    params.loaded = true;
    return true;
}

bool SanaTextEncoderModel::load_weights_from_gguf(ggml_context *ctx_weights_from_gguf, ggml_backend_buffer_t buffer) {
    if (!params.loaded) {
        fprintf(stderr, "Parameters not loaded before weights.\n");
        return false;
    }
    (void)buffer;

    char name_buf[256];
    const char* prefix = "text_encoder.";

    auto load_tensor_meta = [&](const char* suffix, ggml_tensor*& tensor_ptr_member, bool required = true) {
        snprintf(name_buf, sizeof(name_buf), "%s%s", prefix, suffix);
        ggml_tensor* t_meta = ggml_get_tensor(ctx_weights_from_gguf, name_buf);
        if (!t_meta) {
            snprintf(name_buf, sizeof(name_buf), "%s", suffix);
            t_meta = ggml_get_tensor(ctx_weights_from_gguf, name_buf);
        }

        if (t_meta) {
            tensor_ptr_member = t_meta;
            tensors_map[std::string(name_buf)] = t_meta;
        } else if (required) {
            fprintf(stderr, "Error: Required tensor '%s%s' (or '%s') not found in GGUF context.\n", prefix, suffix, suffix);
            return false;
        } else {
            tensor_ptr_member = nullptr;
        }
        return true;
    };

    if (!load_tensor_meta("token_embd.weight", tok_embeddings_weight, true)) return false;
    load_tensor_meta("pos_embd.weight", pos_embeddings_weight, false);

    layers.resize(params.n_layer);
    for (uint32_t i = 0; i < params.n_layer; ++i) {
        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_q.weight", i); if(!load_tensor_meta(name_buf, layers[i].attn_q_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_k.weight", i); if(!load_tensor_meta(name_buf, layers[i].attn_k_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_v.weight", i); if(!load_tensor_meta(name_buf, layers[i].attn_v_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_output.weight", i); if(!load_tensor_meta(name_buf, layers[i].attn_o_w, true)) return false;

        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_norm.weight", i); if(!load_tensor_meta(name_buf, layers[i].attn_norm_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.attn_norm.bias", i); load_tensor_meta(name_buf, layers[i].attn_norm_b, false);

        snprintf(name_buf, sizeof(name_buf), "blk.%u.ffn_gate.weight", i); if(!load_tensor_meta(name_buf, layers[i].ffn_gate_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.ffn_down.weight", i); if(!load_tensor_meta(name_buf, layers[i].ffn_down_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.ffn_up.weight", i); load_tensor_meta(name_buf, layers[i].ffn_up_w, (layers[i].ffn_gate_w !=nullptr) );

        snprintf(name_buf, sizeof(name_buf), "blk.%u.ffn_norm.weight", i); if(!load_tensor_meta(name_buf, layers[i].ffn_norm_w, true)) return false;
        snprintf(name_buf, sizeof(name_buf), "blk.%u.ffn_norm.bias", i); load_tensor_meta(name_buf, layers[i].ffn_norm_b, false);
    }

    if (!load_tensor_meta("final_norm.weight", final_norm_w, true)) return false;
    load_tensor_meta("final_norm.bias", final_norm_b, false);

    return true;
}

void SanaTextEncoderModel::init_weights(ggml_context *ctx_weights, ggml_type wtype) {
    if (!params.loaded) {
        params = SanaTextEncoderParams();
    }

    tok_embeddings_weight = ggml_new_tensor_2d(ctx_weights, wtype, params.n_embd, params.n_vocab);
    ggml_set_name(tok_embeddings_weight, "te.tok_embd.w");
    tensors_map["te.tok_embd.w"] = tok_embeddings_weight;

    bool uses_learned_pos = (params.n_rel_attn_bkts == 0);
    if (uses_learned_pos) {
        pos_embeddings_weight = ggml_new_tensor_2d(ctx_weights, wtype, params.n_embd, params.n_ctx_train);
        ggml_set_name(pos_embeddings_weight, "te.pos_embd.w");
        tensors_map["te.pos_embd.w"] = pos_embeddings_weight;
    } else {
        pos_embeddings_weight = nullptr;
    }

    layers.resize(params.n_layer);
    for (uint32_t i = 0; i < params.n_layer; ++i) {
        char name_buf[128];
        SanaTextEncoderLayer& layer = layers[i];
        auto set_tensor = [&](ggml_tensor*& member, const char* name_suffix, bool required, int64_t d0, int64_t d1 = 0, int64_t d2 = 0, int64_t d3 = 0) {
            (void)required;
            snprintf(name_buf, sizeof(name_buf), "te.blk.%u.%s", i, name_suffix);
            if (d3) member = ggml_new_tensor_4d(ctx_weights, wtype, d0, d1, d2, d3);
            else if (d2) member = ggml_new_tensor_3d(ctx_weights, wtype, d0, d1, d2);
            else if (d1) member = ggml_new_tensor_2d(ctx_weights, wtype, d0, d1);
            else member = ggml_new_tensor_1d(ctx_weights, wtype, d0);
            ggml_set_name(member, name_buf);
            tensors_map[name_buf] = member;
        };
         auto set_norm_weights = [&](ggml_tensor*& w, ggml_tensor*&b, const char* base_name, bool bias_required = false){
            snprintf(name_buf, sizeof(name_buf), "te.blk.%u.%s.weight", i, base_name);
            w = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, params.n_embd); ggml_set_name(w, name_buf); tensors_map[name_buf] = w;
            if (bias_required) {
                snprintf(name_buf, sizeof(name_buf), "te.blk.%u.%s.bias", i, base_name);
                b = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, params.n_embd); ggml_set_name(b, name_buf); tensors_map[name_buf] = b;
            } else {b = nullptr;}
        };

        set_tensor(layer.attn_q_w, "attn_q.weight", true, params.n_embd, params.n_embd);
        set_tensor(layer.attn_k_w, "attn_k.weight", true, params.n_embd, params.n_embd);
        set_tensor(layer.attn_v_w, "attn_v.weight", true, params.n_embd, params.n_embd);
        set_tensor(layer.attn_o_w, "attn_o.weight", true, params.n_embd, params.n_embd);
        set_norm_weights(layer.attn_norm_w, layer.attn_norm_b, "attn_norm", false);

        set_tensor(layer.ffn_gate_w, "ffn_gate.weight", true, params.n_embd, params.n_ff);
        set_tensor(layer.ffn_down_w, "ffn_down.weight", true, params.n_ff, params.n_embd);
        set_tensor(layer.ffn_up_w,   "ffn_up.weight",   false, params.n_embd, params.n_ff);
        set_norm_weights(layer.ffn_norm_w, layer.ffn_norm_b, "ffn_norm", false);
    }

    final_norm_w = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, params.n_embd);
    ggml_set_name(final_norm_w, "te.final_norm.w"); tensors_map["te.final_norm.w"] = final_norm_w;
    final_norm_b = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, params.n_embd);
    ggml_set_name(final_norm_b, "te.final_norm.b"); tensors_map["te.final_norm.b"] = final_norm_b;
}

ggml_cgraph* SanaTextEncoderModel::build_graph(
    ggml_context *ctx,
    ggml_tensor *token_ids,
    ggml_tensor *input_positions
) {
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, SANA_TEXT_ENCODER_GRAPH_SIZE, false);

    ggml_tensor *cur = ggml_get_rows(ctx, tok_embeddings_weight, token_ids);
    ggml_set_name(cur, "token_embeddings");

    if (pos_embeddings_weight && input_positions) {
        ggml_tensor *pos_embeds = ggml_get_rows(ctx, pos_embeddings_weight, input_positions);
        cur = ggml_add(ctx, cur, pos_embeds);
        ggml_set_name(cur, "token_plus_pos_embeddings");
    }

    for (uint32_t i = 0; i < params.n_layer; ++i) {
        SanaTextEncoderLayer& layer = layers[i];
        char name_buf[128];

        ggml_tensor *residual = cur;

        ggml_tensor *ln1_out = ggml_rms_norm(ctx, cur, params.norm_eps);
        ln1_out = ggml_mul(ctx, ln1_out, ggml_reshape_2d(ctx, layer.attn_norm_w, params.n_embd, 1));
        if (layer.attn_norm_b) ln1_out = ggml_add(ctx, ln1_out, ggml_reshape_2d(ctx, layer.attn_norm_b, params.n_embd, 1));
        snprintf(name_buf, sizeof(name_buf), "blk%u.ln1_out", i); ggml_set_name(ln1_out, name_buf);

        ggml_tensor *q = ggml_mul_mat(ctx, layer.attn_q_w, ln1_out);
        ggml_tensor *k = ggml_mul_mat(ctx, layer.attn_k_w, ln1_out);
        ggml_tensor *v = ggml_mul_mat(ctx, layer.attn_v_w, ln1_out);

        q = ggml_reshape_3d(ctx, q, params.n_rot, params.n_head, q->ne[1]);
        k = ggml_reshape_3d(ctx, k, params.n_rot, params.n_head, k->ne[1]);
        v = ggml_reshape_3d(ctx, v, params.n_rot, params.n_head, v->ne[1]);

        if (params.n_rot > 0 && input_positions) {
             q = ggml_rope_ext(ctx, q, input_positions, nullptr, params.n_rot, 0 , params.n_ctx_train, params.rope_freq_base, params.rope_freq_scale, 0.0f,0.0f,0.0f,0.0f);
             k = ggml_rope_ext(ctx, k, input_positions, nullptr, params.n_rot, 0 , params.n_ctx_train, params.rope_freq_base, params.rope_freq_scale, 0.0f,0.0f,0.0f,0.0f);
        }

        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        ggml_tensor *attn_scores = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f/sqrtf((float)params.n_rot), 0.0f, 0);
        attn_scores = ggml_permute(ctx, attn_scores, 0, 2, 1, 3);
        attn_scores = ggml_reshape_2d(ctx, attn_scores, params.n_embd, attn_scores->ne[2]);

        ggml_tensor *attn_out = ggml_mul_mat(ctx, layer.attn_o_w, attn_scores);
        snprintf(name_buf, sizeof(name_buf), "blk%u.attn_out",i); ggml_set_name(attn_out, name_buf);

        cur = ggml_add(ctx, residual, attn_out);
        snprintf(name_buf, sizeof(name_buf), "blk%u.attn_add_resid",i); ggml_set_name(cur, name_buf);

        residual = cur;
        ggml_tensor *ln2_out = ggml_rms_norm(ctx, cur, params.norm_eps);
        ln2_out = ggml_mul(ctx, ln2_out, ggml_reshape_2d(ctx, layer.ffn_norm_w, params.n_embd, 1));
        if (layer.ffn_norm_b) ln2_out = ggml_add(ctx, ln2_out, ggml_reshape_2d(ctx, layer.ffn_norm_b,params.n_embd,1));
        snprintf(name_buf, sizeof(name_buf), "blk%u.ln2_out",i); ggml_set_name(ln2_out, name_buf);

        ggml_tensor *ffn_hidden = ggml_mul_mat(ctx, layer.ffn_gate_w, ln2_out);
        if(layer.ffn_gate_b) ffn_hidden = ggml_add(ctx, ffn_hidden, ggml_reshape_2d(ctx, layer.ffn_gate_b, params.n_ff,1));

        if (layer.ffn_up_w) {
            ggml_tensor *ffn_up_val = ggml_mul_mat(ctx, layer.ffn_up_w, ln2_out);
            if(layer.ffn_up_b) ffn_up_val = ggml_add(ctx, ffn_up_val, ggml_reshape_2d(ctx, layer.ffn_up_b, params.n_ff,1));
            ffn_hidden = ggml_mul(ctx, ffn_hidden, ggml_silu(ctx, ffn_up_val));
        } else {
            ffn_hidden = ggml_silu(ctx, ffn_hidden);
        }
        snprintf(name_buf, sizeof(name_buf), "blk%u.ffn_hidden",i); ggml_set_name(ffn_hidden, name_buf);

        ggml_tensor *ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_hidden);
        if(layer.ffn_down_b) ffn_out = ggml_add(ctx, ffn_out, ggml_reshape_2d(ctx, layer.ffn_down_b, params.n_embd,1));
        snprintf(name_buf, sizeof(name_buf), "blk%u.ffn_out",i); ggml_set_name(ffn_out, name_buf);

        cur = ggml_add(ctx, residual, ffn_out);
        snprintf(name_buf, sizeof(name_buf), "blk%u.ffn_add_resid",i); ggml_set_name(cur, name_buf);
    }

    cur = ggml_rms_norm(ctx, cur, params.norm_eps);
    cur = ggml_mul(ctx, cur, ggml_reshape_2d(ctx, final_norm_w, params.n_embd, 1));
    if (final_norm_b) cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, final_norm_b, params.n_embd,1));
    ggml_set_name(cur, "final_norm_out");

    ggml_build_forward_expand(gf, cur);
    return gf;
}
