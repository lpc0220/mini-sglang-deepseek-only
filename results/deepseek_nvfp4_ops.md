# DeepSeek-R1-NVFP4-v2 Execution Ops

**Model:** [nvidia/DeepSeek-R1-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-NVFP4-v2)

**Generated:** 2026-01-23

---

## Model Configuration

| Parameter | Value | Symbol |
|-----------|-------|--------|
| `hidden_size` | 7168 | H |
| `q_lora_rank` | 1536 | Lq |
| `kv_lora_rank` | 512 | Lkv |
| `num_attention_heads` | 128 | Nh |
| `num_key_value_heads` | 128 | Nkv |
| `qk_nope_head_dim` | 128 | Dn |
| `qk_rope_head_dim` | 64 | Dr |
| `v_head_dim` | 128 | Dv |
| `qk_head_dim` | 192 | Dq = Dn + Dr |
| `moe_intermediate_size` | 2048 | I |
| `num_experts_per_tok` | 8 | K |
| `n_routed_experts` | 256 | E |
| `n_shared_experts` | 1 | - |
| `num_hidden_layers` | 61 | L |
| `first_k_dense_replace` | 3 | - |

**Shape Variables:**
- `B` = batch size (number of samples/requests)
- `S` = sequence length (prefill only; decode has S=1)

---

## Execution Path

### Prefill Phase (MHA path)

```
self.model()
└─> layer()
    ├─> self.input_layernorm()                        [B*S, 7168] → [B*S, 7168]
    │
    ├─> self.self_attn()
    │   ├─> self.forward_normal_prepare()
    │   │   ├─> self.fused_qkv_a_proj_with_mqa()      [B*S, 7168] → [B*S, 2112] → split to q:[B*S, 1536], kv_a:[B*S, 512], k_pe:[B*S, 64]
    │   │   ├─> self.q_a_layernorm()                  [B*S, 1536] → [B*S, 1536]
    │   │   ├─> self.q_b_proj()                       [B*S, 1536] → [B*S, 24576] → view [B*S, 128, 192]
    │   │   ├─> self.kv_a_layernorm()                 [B*S, 512] → [B*S, 512]
    │   │   ├─> self.kv_b_proj()                      [B*S, 512] → [B*S, 32768] → view [B*S, 128, 256] → split k_nope:[B*S, 128, 128], v:[B*S, 128, 128]
    │   │   ├─> self.rotary_emb()                     q_pe:[B*S, 128, 64], k_pe:[B*S, 1, 64]
    │   │   └─> self._concat_and_cast_mha_k()         k_nope:[B*S, 128, 128], k_pe:[B*S, 1, 64] → [B*S, 128, 192]
    │   │
    │   └─> self.forward_normal_core()
    │       ├─> self.attn_mha()                       q:[B*S, 128, 192], k:[B*S, 128, 192], v:[B*S, 128, 128] → [B*S, 128, 128] → reshape [B*S, 16384]
    │       └─> self.o_proj()                         [B*S, 16384] → [B*S, 7168]
    │
    ├─> self.post_attention_layernorm()               [B*S, 7168] → [B*S, 7168]
    │
    └─> self.mlp()
        ├─> **Dense MLP (layers 0-2)**
        │   ├─> self.gate_up_proj()                   [B*S, 7168] → [B*S, 4096]
        │   ├─> self.act_fn()                         [B*S, 4096] → [B*S, 2048]
        │   └─> self.down_proj()                      [B*S, 2048] → [B*S, 7168]
        │
        └─> **MoE MLP (layers 3-60)**
            ├─> self.gate()                           [B*S, 7168] → [B*S, 256]
            ├─> self.topk()                           [B*S, 256] → weights:[B*S, 8], ids:[B*S, 8]
            ├─> self.shared_experts()                 [B*S, 7168] → [B*S, 7168]
            └─> self.experts()                        [B*S, 7168] → [B*S, 7168]
```

### Decode Phase (MLA/Absorb path)

```
self.model()
└─> layer()
    ├─> self.input_layernorm()                        [B, 7168] → [B, 7168]
    │
    ├─> self.self_attn()
    │   ├─> self.forward_absorb_prepare()
    │   │   ├─> self.fused_qkv_a_proj_with_mqa()      [B, 7168] → [B, 2112] → split to q:[B, 1536], kv_a:[B, 512], k_pe:[B, 64]
    │   │   ├─> self.q_a_layernorm()                  [B, 1536] → [B, 1536]
    │   │   ├─> self.kv_a_layernorm()                 [B, 512] → [B, 512]
    │   │   ├─> self.q_b_proj()                       [B, 1536] → [B, 24576] → view [B, 128, 192] → split q_nope:[B, 128, 128], q_pe:[B, 128, 64]
    │   │   ├─> bmm_fp8() (q_nope * w_kc)             q_nope:[128, B, 128] @ w_kc:[128, 128, 512] → [128, B, 512] → transpose [B, 128, 512]
    │   │   └─> self.rotary_emb()                     q_pe:[B, 128, 64], k_pe:[B, 1, 64]
    │   │
    │   └─> self.forward_absorb_core()
    │       ├─> self.attn_mqa()                       q_nope:[B, 128, 512], k_nope:[B, 1, 512], q_pe:[B, 128, 64], k_pe:[B, 1, 64] → [B, 128, 512] → view [128, B, 512]
    │       ├─> bmm_fp8() (attn * w_vc)               attn:[128, B, 512] @ w_vc:[128, 512, 128] → [128, B, 128] → transpose [B, 128, 128] → reshape [B, 16384]
    │       └─> self.o_proj()                         [B, 16384] → [B, 7168]
    │
    ├─> self.post_attention_layernorm()               [B, 7168] → [B, 7168]
    │
    └─> self.mlp()
        ├─> **Dense MLP (layers 0-2)**
        │   ├─> self.gate_up_proj()                   [B, 7168] → [B, 4096]
        │   ├─> self.act_fn()                         [B, 4096] → [B, 2048]
        │   └─> self.down_proj()                      [B, 2048] → [B, 7168]
        │
        └─> **MoE MLP (layers 3-60)**
            ├─> self.gate()                           [B, 7168] → [B, 256]
            ├─> self.topk()                           [B, 256] → weights:[B, 8], ids:[B, 8]
            ├─> self.shared_experts()                 [B, 7168] → [B, 7168]
            └─> self.experts()                        [B, 7168] → [B, 7168]
```

### Speculative Decoding (EAGLE/NextN)

```
DeepseekV3ForCausalLMNextN.forward()
└─> self.model() (DeepseekModelNextN)
    ├─> self.embed_tokens()                           [B] → [B, 7168]
    ├─> self.enorm()                                  [B, 7168] → [B, 7168]
    ├─> self.hnorm()                                  [B, 7168] → [B, 7168]
    ├─> self.eh_proj()                                [B, 14336] → [B, 7168]
    │
    └─> self.decoder() **(Single DeepseekV2DecoderLayer)**
        ├─> (same ops as decode phase above)
        └─> ...
    │
    └─> self.shared_head.norm()                       [B, 7168] → [B, 7168]
```

---

## Prefill Phase (MHA path)

Uses `forward_normal_prepare` + `forward_normal_core`.

### Attention Ops

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.input_layernorm()` | [B*S, H] | [B*S, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.fused_qkv_a_proj_with_mqa()` | [B*S, H] | [B*S, Lq + Lkv + Dr] = [B*S, 2112] | `cutlass_scaled_fp4_mm` |
| `self.q_a_layernorm()` | [B*S, Lq] = [B*S, 1536] | [B*S, Lq] | `rmsnorm` |
| `self.q_b_proj()` | [B*S, Lq] | [B*S, Nh * Dq] = [B*S, 24576] | `cutlass_scaled_fp4_mm` |
| `self.kv_a_layernorm()` | [B*S, Lkv] = [B*S, 512] | [B*S, Lkv] | `rmsnorm` |
| `self.kv_b_proj()` | [B*S, Lkv] | [B*S, Nh * (Dn + Dv)] = [B*S, 32768] | `cutlass_scaled_fp4_mm` |
| `self.rotary_emb()` | q_pe: [B*S, Nh, Dr], k_pe: [B*S, 1, Dr] | same | `apply_rope_with_cos_sin_cache_inplace` |
| `self._concat_and_cast_mha_k()` | k_nope: [B*S, Nh, Dn], k_pe: [B*S, 1, Dr] | [B*S, Nh, Dn + Dr] | `concat_mla_mha_k` |
| `self.attn_mha()` | q: [B*S, Nh, Dq], k: [B*S, Nh, Dq], v: [B*S, Nh, Dv] | [B*S, Nh, Dv] | flashinfer/trtllm prefill |
| `self.o_proj()` | [B*S, Nh * Dv] = [B*S, 16384] | [B*S, H] | `cutlass_scaled_fp4_mm` |

### MLP Ops (MoE, layers 3-60)

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.post_attention_layernorm()` | [B*S, H] | [B*S, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.gate()` | [B*S, H] | [B*S, E] = [B*S, 256] | `dsv3_router_gemm` / `cutlass_scaled_fp4_mm` |
| `self.topk()` | [B*S, E] | topk_weights: [B*S, K], topk_ids: [B*S, K] | `topk_softmax` / `topk_sigmoid` / `moe_fused_gate` |
| `self.shared_experts()` | [B*S, H] | [B*S, H] | `cutlass_scaled_fp4_mm` + `silu_and_mul` |
| `self.experts()` | [B*S, H] | [B*S, H] | (varies by backend) |

### MLP Ops (Dense, layers 0-2)

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.post_attention_layernorm()` | [B*S, H] | [B*S, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.gate_up_proj()` | [B*S, H] | [B*S, 2 * I] = [B*S, 4096] | `cutlass_scaled_fp4_mm` |
| `self.act_fn()` | [B*S, 2 * I] | [B*S, I] | `silu_and_mul` |
| `self.down_proj()` | [B*S, I] | [B*S, H] | `cutlass_scaled_fp4_mm` |

---

## Decode Phase (MLA/Absorb path)

Uses `forward_absorb_prepare` + `forward_absorb_core`. For decode, `S=1`, so shapes are `[B, ...]`.

### Attention Ops

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.input_layernorm()` | [B, H] | [B, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.fused_qkv_a_proj_with_mqa()` | [B, H] | [B, Lq + Lkv + Dr] = [B, 2112] | `dsv3_fused_a_gemm` (B<=16) / `cutlass_scaled_fp4_mm` |
| `self.q_a_layernorm()` | [B, Lq] = [B, 1536] | [B, Lq] | `rmsnorm` |
| `self.kv_a_layernorm()` | [B, Lkv] = [B, 512] | [B, Lkv] | `rmsnorm` |
| `self.q_b_proj()` | [B, Lq] | [B, Nh * Dq] = [B, 24576] | `cutlass_scaled_fp4_mm` |
| `bmm_fp8()` (q_nope * w_kc) | q_nope: [Nh, B, Dn], w_kc: [Nh, Dn, Lkv] | [Nh, B, Lkv] | `bmm_fp8` |
| `self.rotary_emb()` | q_pe: [B, Nh, Dr], k_pe: [B, 1, Dr] | same | `apply_rope_with_cos_sin_cache_inplace` |
| `self.attn_mqa()` | q: [B, Nh, Lkv+Dr], k_nope: [B, 1, Lkv], k_pe: [B, 1, Dr] | [B, Nh, Lkv] | `cutlass_mla_decode` / `trtllm_batch_decode_with_kv_cache_mla` |
| `bmm_fp8()` (attn_output * w_vc) | attn: [Nh, B, Lkv], w_vc: [Nh, Lkv, Dv] | [Nh, B, Dv] | `bmm_fp8` |
| `self.o_proj()` | [B, Nh * Dv] = [B, 16384] | [B, H] | `cutlass_scaled_fp4_mm` |

### MLP Ops (MoE, layers 3-60)

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.post_attention_layernorm()` | [B, H] | [B, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.gate()` | [B, H] | [B, E] = [B, 256] | `dsv3_router_gemm` (B<=16) / `cutlass_scaled_fp4_mm` |
| `self.topk()` | [B, E] | topk_weights: [B, K], topk_ids: [B, K] | `topk_softmax` / `topk_sigmoid` / `moe_fused_gate` |
| `self.shared_experts()` | [B, H] | [B, H] | `cutlass_scaled_fp4_mm` + `silu_and_mul` |
| `self.experts()` | [B, H] | [B, H] | (varies by backend) |

### MLP Ops (Dense, layers 0-2)

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.post_attention_layernorm()` | [B, H] | [B, H] | `rmsnorm` / `fused_add_rmsnorm` |
| `self.gate_up_proj()` | [B, H] | [B, 2 * I] = [B, 4096] | `cutlass_scaled_fp4_mm` |
| `self.act_fn()` | [B, 2 * I] | [B, I] | `silu_and_mul` |
| `self.down_proj()` | [B, I] | [B, H] | `cutlass_scaled_fp4_mm` |

---

## MoE Expert Backends

### Cutlass Backend

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `prepare_moe_input` | [B, H], topk_ids: [B, K] | permuted: [B*K, H] | `prepare_moe_input` |
| `scaled_fp4_experts_quant` | [B*K, H] | quantized input | `scaled_fp4_experts_quant` |
| `cutlass_fp4_group_mm` (gate_up) | [B*K, H] | [B*K, 2*I] | `cutlass_fp4_group_mm` |
| `silu_and_mul` | [B*K, 2*I] | [B*K, I] | `silu_and_mul` |
| `cutlass_fp4_group_mm` (down) | [B*K, I] | [B*K, H] | `cutlass_fp4_group_mm` |
| `apply_shuffle_mul_sum` | [B*K, H], topk_weights | [B, H] | `apply_shuffle_mul_sum` |

### TRTLLM Backend

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| Fused MoE | [B, H], topk_ids, topk_weights | [B, H] | `trtllm_fp4_block_scale_moe` |

### Triton Backend

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| Fused MoE | [B, H], topk_ids, topk_weights | [B, H] | `fused_moe_kernel` |

---

## Speculative Decoding (EAGLE/NextN)

Uses `DeepseekV3ForCausalLMNextN` - a separate model with distinct architecture.

### NextN-specific Ops

| Op | Input Shape | Output Shape | Kernel |
|----|-------------|--------------|--------|
| `self.embed_tokens()` | [B] | [B, H] | embedding lookup |
| `self.enorm()` | [B, H] (embeddings) | [B, H] | `rmsnorm` |
| `self.hnorm()` | [B, H] (target hidden) | [B, H] | `rmsnorm` |
| `self.eh_proj()` | [B, 2*H] | [B, H] | linear (not FP4) |
| `self.shared_head.norm()` | [B, H] | [B, H] | `rmsnorm` |

### Decoder Layer (single layer)

Same ops as main model decode phase - uses `forward_absorb_prepare` + `forward_absorb_core`.

Key differences:
- **Single decoder layer** (vs 61 layers in main model)
- **eh_proj**: Combines embeddings + target hidden states
- **enorm/hnorm**: Extra RMSNorm layers for input processing

---

## Shape Derivations

For nvidia/DeepSeek-R1-NVFP4-v2:

**GEMM vs BMM:**
- **GEMM** (2D): Linear projections like `q_b_proj` - shape `[B, 1536] @ [1536, 24576] → [B, 24576]`
- **BMM** (3D): Absorbed MLA matmuls like `q_nope * w_kc` - shape `[128, B, 128] @ [128, 128, 512] → [128, B, 512]` (batched over heads)

```
H = 7168 (hidden_size)
Nh = 128 (num_heads)
Lq = 1536 (q_lora_rank)
Lkv = 512 (kv_lora_rank)
Dn = 128 (qk_nope_head_dim)
Dr = 64 (qk_rope_head_dim)
Dv = 128 (v_head_dim)
Dq = Dn + Dr = 192 (qk_head_dim)
E = 256 (n_routed_experts)
K = 8 (num_experts_per_tok)
I = 2048 (moe_intermediate_size)

# Derived dimensions
fused_qkv_a_proj output: Lq + Lkv + Dr = 1536 + 512 + 64 = 2112
q_b_proj output: Nh * Dq = 128 * 192 = 24576
kv_b_proj output: Nh * (Dn + Dv) = 128 * (128 + 128) = 32768
o_proj input: Nh * Dv = 128 * 128 = 16384
```

---

## Kernel Summary

| Kernel | Source | Ops Using It |
|--------|--------|--------------|
| `rmsnorm` | sgl-kernel | input_layernorm, post_attention_layernorm, q_a_layernorm, kv_a_layernorm, enorm, hnorm |
| `fused_add_rmsnorm` | sgl-kernel | input_layernorm, post_attention_layernorm |
| `cutlass_scaled_fp4_mm` | sgl-kernel | fused_qkv_a_proj, q_b_proj, kv_b_proj, o_proj, gate_up_proj, down_proj, gate, shared_experts |
| `dsv3_fused_a_gemm` | sgl-kernel | fused_qkv_a_proj_with_mqa (low-latency, B<=16) |
| `dsv3_router_gemm` | sgl-kernel | gate (low-latency, B<=16) |
| `bmm_fp8` | sgl-kernel | q_nope * w_kc, attn_output * w_vc |
| `apply_rope_with_cos_sin_cache_inplace` | sgl-kernel | rotary_emb |
| `concat_mla_mha_k` | sgl-kernel | _concat_and_cast_mha_k |
| `silu_and_mul` | sgl-kernel | act_fn |
| `cutlass_mla_decode` | sgl-kernel | attn_mqa (decode) |
| `trtllm_batch_decode_with_kv_cache_mla` | flashinfer | attn_mqa (decode, trtllm backend) |
| `trtllm_ragged_attention_deepseek` | flashinfer | attn_mqa (prefill, trtllm backend) |
| `mla_rope_quantize_fp8` | flashinfer | attn_mqa (trtllm fp8) |
| `topk_softmax` / `topk_sigmoid` / `moe_fused_gate` | sgl-kernel | topk |
| `prepare_moe_input` | sgl-kernel | experts (cutlass) |
| `scaled_fp4_experts_quant` | sgl-kernel | experts (cutlass) |
| `cutlass_fp4_group_mm` | sgl-kernel | experts (cutlass) |
| `apply_shuffle_mul_sum` | sgl-kernel | experts (cutlass) |
| `moe_align_block_size` | sgl-kernel | experts (cutlass) |
| `trtllm_fp4_block_scale_moe` | flashinfer | experts (trtllm) |
| `fused_moe_kernel` | triton | experts (triton) |
