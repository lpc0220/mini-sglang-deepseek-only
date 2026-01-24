# DeepSeek Kernel Traces

This document traces all kernel calls from DeepSeek models back to their source definitions, following the call chain format similar to GDB's `bt` (backtrace) command.

**Generated:** 2026-01-20

**Format:** Each line shows a **call site** (where the function is called), not where it's defined.

---

## Summary Table

| Kernel | Source | Category | Primary Call Site |
|--------|--------|----------|-------------------|
| `cutlass_mla_decode` | sgl-kernel | Attention | cutlass_mla_backend.py:274 |
| `cutlass_mla_get_workspace_size` | sgl-kernel | Attention | cutlass_mla_backend.py:107 |
| `silu_and_mul` | sgl-kernel | Activation | activation.py:58 |
| `gelu_and_mul` | sgl-kernel | Activation | activation.py:74 |
| `gelu_tanh_and_mul` | sgl-kernel | Activation | activation.py:72 |
| `rmsnorm` | sgl-kernel | Normalization | layernorm.py:102 |
| `fused_add_rmsnorm` | sgl-kernel | Normalization | layernorm.py:100 |
| `int8_scaled_mm` | sgl-kernel | GEMM | w8a8_int8.py:198 |
| `fp8_scaled_mm` | sgl-kernel | GEMM | fp8_utils.py:909 |
| `fp8_blockwise_scaled_mm` | sgl-kernel | GEMM | fp8_utils.py:333 |
| `fp8_blockwise_scaled_grouped_mm` | sgl-kernel | MoE GEMM | cutlass_moe.py:177 |
| `cutlass_scaled_fp4_mm` | sgl-kernel | GEMM | modelopt_quant.py:135 |
| `topk_softmax` | sgl-kernel | MoE Gating | topk.py:404 |
| `topk_sigmoid` | sgl-kernel | MoE Gating | topk.py:411 |
| `moe_fused_gate` | sgl-kernel | MoE Gating | topk.py:643 |
| `moe_align_block_size` | sgl-kernel | MoE | moe_align_block_size.py:74 |
| `moe_sum_reduce` | sgl-kernel | MoE | fused_moe.py:551 |
| `apply_rope_with_cos_sin_cache_inplace` | sgl-kernel | RoPE | rotary_embedding.py:242 |
| `rotary_embedding` | sgl-kernel | RoPE | rotary_embedding.py:96 |
| `merge_state` | sgl-kernel | Attention | flashinfer_backend.py:784 |
| `bmm_fp8` | sgl-kernel | GEMM | deepseek_v2.py:1521 |
| `dsv3_fused_a_gemm` | sgl-kernel | GEMM | deepseek_v2.py:1296 |
| `dsv3_router_gemm` | sgl-kernel | MoE Gating | deepseek_v2.py:330 |
| `merge_state_v2` | sgl-kernel | Attention | deepseek_v2.py:1886 |
| `concat_mla_k` | sgl-kernel | Attention | deepseek_v2.py:2046 |
| `concat_mla_absorb_q` | sgl-kernel | Attention | trtllm_mla_backend.py:1215 |
| `prepare_moe_input` | sgl-kernel | MoE | cutlass_moe.py:140 |
| `shuffle_rows` | sgl-kernel | MoE | cutlass_moe.py:153 |
| `es_fp8_blockwise_scaled_grouped_mm` | sgl-kernel | MoE GEMM | cutlass_moe.py:163 |
| `apply_shuffle_mul_sum` | sgl-kernel | MoE | cutlass_moe.py:242 |
| `scaled_fp4_experts_quant` | sgl-kernel | MoE | cutlass_moe.py:353 |
| `cutlass_fp4_group_mm` | sgl-kernel | MoE GEMM | cutlass_moe.py:361 |
| `cutlass_w4a8_moe_mm` | sgl-kernel | MoE GEMM | cutlass_w4a8_moe.py:155 |
| `get_cutlass_w4a8_moe_mm_data` | sgl-kernel | MoE | cutlass_w4a8_moe.py:140 |
| `sgl_per_token_quant_fp8` | sgl-kernel | Quantization | fp8_kernel.py:544 |
| `sgl_per_token_group_quant_fp8` | sgl-kernel | Quantization | fp8_kernel.py:480 |
| `sgl_per_token_group_quant_8bit` | sgl-kernel | Quantization | int8_kernel.py:218 |
| `sgl_per_token_group_quant_int8` | sgl-kernel | Quantization | int8_kernel.py:223 |
| `qserve_w4a8_per_chn_gemm` | sgl-kernel | GEMM | qoq.py:230 |
| `qserve_w4a8_per_group_gemm` | sgl-kernel | GEMM | qoq.py:235 |
| `hadamard_transform` | sgl-kernel | NSA | nsa_indexer.py:105 |
| `fast_topk` | sgl-kernel | Speculative | spec_utils.py, eagle_worker.py |
| `fast_topk_v2` | sgl-kernel | NSA | nsa_backend.py:211 |
| `fast_topk_transform_fused` | sgl-kernel | NSA | nsa_backend.py:214 |
| `fast_topk_transform_ragged_fused` | sgl-kernel | NSA | nsa_backend.py:223 |
| `min_p_sampling_from_probs` | sgl-kernel | Sampling | sampler.py:133 |
| `top_k_renorm_prob` | sgl-kernel | Sampling | sampler.py:131 |
| `top_p_renorm_prob` | sgl-kernel | Sampling | sampler.py:132 |
| `top_k_top_p_sampling_from_probs` | sgl-kernel | Sampling | sampler.py:137 |
| `build_tree_kernel_efficient` | sgl-kernel | Speculative | eagle_utils.py, eagle_worker.py |
| `verify_tree_greedy` | sgl-kernel | Speculative | eagle_utils.py:157 |
| `tree_speculative_sampling_target_only` | sgl-kernel | Speculative | eagle_info.py:350 |
| `weak_ref_tensor` | sgl-kernel | Utility | weak_ref_tensor.py:21 |
| `trtllm_batch_decode_with_kv_cache_mla` | flashinfer | Attention | trtllm_mla_backend.py:925 |
| `trtllm_ragged_attention_deepseek` | flashinfer | Attention | trtllm_mla_backend.py:1166 |
| `trtllm_bf16_moe` | flashinfer | MoE | layer.py:1045 |
| `trtllm_fp8_block_scale_moe` | flashinfer | MoE | flashinfer_trtllm.py:168 |
| `trtllm_fp4_block_scale_moe` | flashinfer | MoE | layer.py:1188 |
| `mla_rope_quantize_fp8` | flashinfer | RoPE | trtllm_mla_backend.py:726 |
| `fused_moe_kernel` | triton | MoE | fused_moe_triton_kernels.py:770 |

---

## Kernel → Ops Mapping Table

This table shows which ops (in DeepSeek model code) use each kernel. Generic kernels like GEMM can map to multiple ops.

| Kernel | Target Ops | Op Call Sites (deepseek_v2.py) |
|--------|------------|--------------------------------|
| `fp8_blockwise_scaled_mm` | `fused_qkv_a_proj_with_mqa`, `q_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, `kv_b_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `shared_experts` | :1300, :1340, :1343, :1324, :1337, :1372, :1383, :1463, :1476, :1486, :1489, :1661, :1687, :1691, :1693, :1696, :1843, :271, :273, :555 |
| `int8_scaled_mm` | `fused_qkv_a_proj_with_mqa`, `q_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, `kv_b_proj`, `o_proj`, `gate_up_proj`, `down_proj` | (same as fp8_blockwise_scaled_mm) |
| `rmsnorm` | `input_layernorm`, `q_a_layernorm`, `kv_a_layernorm`, `post_attention_layernorm` | :1323, :1336, :1348, :1440, :1442, :1445, :1446, :1491, :1690, :1712 |
| `fused_add_rmsnorm` | `input_layernorm`, `post_attention_layernorm` | (via communicator.py) |
| `apply_rope_with_cos_sin_cache_inplace` | `rotary_emb` | :1354, :1533, :1718 |
| `cutlass_mla_decode` | `attn_mqa` | :1576, :1594 |
| `trtllm_batch_decode_with_kv_cache_mla` | `attn_mqa` | :1576, :1594 |
| `trtllm_ragged_attention_deepseek` | `attn_mqa` | :1576, :1594 |
| `mla_rope_quantize_fp8` | `attn_mqa` | :1576, :1594 |
| `merge_state` | `attn_mha` | :1381 |
| `merge_state_v2` | `_chunked_prefix_attn_mha` | :1886 |
| `bmm_fp8` | `forward_absorb_prepare`, `forward_absorb_core`, `forward_absorb_fused_mla_rope_prepare`, `forward_absorb_fused_mla_rope_core` | :1521, :1632, :1705, :1833 |
| `dsv3_fused_a_gemm` | `_forward_fused_qkv_a_proj_with_mqa` | :1296 |
| `dsv3_router_gemm` | `gate (ReplicatedLinear)` | :330 |
| `concat_mla_k` | `_concat_and_cast_mha_k` | :2046 |
| `concat_mla_absorb_q` | `_merge_q` | :1215 (trtllm_mla_backend.py) |
| `prepare_moe_input` | `experts` (cutlass MoE path) | :140 (cutlass_moe.py) |
| `shuffle_rows` | `experts` (cutlass MoE path) | :153 (cutlass_moe.py) |
| `es_fp8_blockwise_scaled_grouped_mm` | `experts` (cutlass MoE path) | :163 (cutlass_moe.py) |
| `apply_shuffle_mul_sum` | `experts` (cutlass MoE path) | :242 (cutlass_moe.py) |
| `scaled_fp4_experts_quant` | `experts` (cutlass FP4 MoE path) | :353 (cutlass_moe.py) |
| `cutlass_fp4_group_mm` | `experts` (cutlass FP4 MoE path) | :361 (cutlass_moe.py) |
| `silu_and_mul` | `act_fn`, `experts`, `shared_experts` | :272, :555, :563 |
| `gelu_and_mul` | `act_fn`, `experts` | activation.py:74, fused_moe.py:475 |
| `gelu_tanh_and_mul` | `act_fn` | activation.py:72 |
| `topk_softmax` | `gate` | :561, :593 |
| `topk_sigmoid` | `gate` | :561, :593 |
| `moe_fused_gate` | `gate` | :561, :593 |
| `fused_moe_kernel` | `experts` | :563 |
| `fp8_blockwise_scaled_grouped_mm` | `experts` | :563 |
| `moe_align_block_size` | `experts` | :563 |
| `moe_sum_reduce` | `experts` (triton MoE path) | fused_moe.py:551 |
| `cutlass_w4a8_moe_mm` | `experts` (W4A8 MoE path) | cutlass_w4a8_moe.py:155 |
| `get_cutlass_w4a8_moe_mm_data` | `experts` (W4A8 MoE path) | cutlass_w4a8_moe.py:140 |
| `trtllm_bf16_moe` | `experts` | :563 |
| `trtllm_fp8_block_scale_moe` | `experts` | :563 |
| `trtllm_fp4_block_scale_moe` | `experts` | :563 |
| `fp8_scaled_mm` | Linear layers (FP8 path) | fp8_utils.py:909 |
| `cutlass_scaled_fp4_mm` | Linear layers (FP4 path) | modelopt_quant.py:135 |
| `qserve_w4a8_per_chn_gemm` | Linear layers (QoQ path) | qoq.py:230 |
| `qserve_w4a8_per_group_gemm` | Linear layers (QoQ path) | qoq.py:235 |
| `sgl_per_token_quant_fp8` | input quantization (FP8) | fp8_kernel.py:544 |
| `sgl_per_token_group_quant_fp8` | input quantization (FP8 grouped) | fp8_kernel.py:480 |
| `sgl_per_token_group_quant_8bit` | input quantization (8-bit grouped) | int8_kernel.py:218 |
| `sgl_per_token_group_quant_int8` | input quantization (INT8 grouped) | int8_kernel.py:223 |
| `rotary_embedding` | `rotary_emb` (fallback path) | rotary_embedding.py:260 |
| `hadamard_transform` | NSA indexer | nsa_indexer.py:105 |
| `fast_topk_v2` | NSA sparse attention | nsa_backend.py:211 |
| `fast_topk_transform_fused` | NSA sparse attention | nsa_backend.py:214 |
| `fast_topk_transform_ragged_fused` | NSA sparse attention | nsa_backend.py:223 |
| `min_p_sampling_from_probs` | Sampling | sampler.py:133 |
| `top_k_renorm_prob` | Sampling | sampler.py:131 |
| `top_p_renorm_prob` | Sampling | sampler.py:132 |
| `top_k_top_p_sampling_from_probs` | Sampling | sampler.py:137 |
| `fast_topk` | Speculative decoding | eagle_worker.py:649, spec_utils.py |
| `build_tree_kernel_efficient` | Speculative decoding | eagle_utils.py, eagle_worker.py |
| `verify_tree_greedy` | Speculative decoding | eagle_utils.py:157, ngram_info.py:293 |
| `tree_speculative_sampling_target_only` | Speculative decoding | eagle_info.py:350 |

---

## Detailed Call Chains

### 1. Attention Kernels

#### 1.1 cutlass_mla_decode **(sgl-kernel)**

MLA (Multi-head Latent Attention) decode kernel for DeepSeek V2/V3.

[cutlass_mla_decode()](../sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py#L274) **(sgl-kernel)**
└─> [forward_decode()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L90)
    └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
        └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
            └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.2 trtllm_batch_decode_with_kv_cache_mla **(flashinfer)**

TRTLLM MLA decode kernel for DeepSeek.

[trtllm_batch_decode_with_kv_cache_mla()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L925) **(flashinfer)**
└─> [forward_decode()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L90)
    └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
        └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
            └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.3 trtllm_ragged_attention_deepseek **(flashinfer)**

TRTLLM prefill attention kernel for DeepSeek.

[trtllm_ragged_attention_deepseek()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L1166) **(flashinfer)**
└─> [forward_extend()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L100)
    └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
        └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
            └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.4 merge_state **(sgl-kernel)**

Merge attention states for chunked attention.

[merge_state()](../sglang/python/sglang/srt/layers/attention/flashinfer_backend.py#L784) **(sgl-kernel)**
└─> [forward_extend()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L100)
    └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
        └─> [self.attn_mha()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1381)
            └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.5 merge_state_v2 **(sgl-kernel)**

Merge attention states for chunked prefix attention (MHA path).

[merge_state_v2()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1886) **(sgl-kernel)**
└─> [self._chunked_prefix_attn_mha()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1929)
    └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
        └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
            └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.6 concat_mla_k **(sgl-kernel)**

Concatenate K nope and K rope for MLA attention.

[concat_mla_k()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2046) **(sgl-kernel)**
└─> [self._concat_and_cast_mha_k()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1377)
    └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
        └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
            └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.7 concat_mla_absorb_q **(sgl-kernel)**

Concatenate Q nope and Q rope for MLA TRTLLM attention.

[concat_mla_absorb_q()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L1215) **(sgl-kernel)**
├─> [_concat_mla_absorb_q_general()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L876) **(forward_decode)**
│   └─> [self.forward_decode()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L90)
│       └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
│           └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
│               └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
│                   └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│                       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│                           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
└─> [_concat_mla_absorb_q_general()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L1003) **(forward_extend)**
    └─> [self.forward_extend()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L100)
        └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
            └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
                └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.8 bmm_fp8 **(sgl-kernel)**

FP8 batched matrix multiplication for MLA weight absorption.

**Path 1: forward_absorb_prepare (q_nope * w_kc)**

[bmm_fp8()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1521) **(sgl-kernel)**
└─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 2: forward_absorb_core (attn_output * w_vc)**

[bmm_fp8()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1632) **(sgl-kernel)**
└─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 3: forward_absorb_fused_mla_rope_prepare (q_nope * w_kc)**

[bmm_fp8()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1705) **(sgl-kernel)**
└─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 4: forward_absorb_fused_mla_rope_core (attn_output * w_vc)**

[bmm_fp8()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1833) **(sgl-kernel)**
└─> [self.forward_absorb_fused_mla_rope_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1282)
    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.9 dsv3_fused_a_gemm **(sgl-kernel)**

Fused QKV A-projection GEMM for DeepSeek V3 low-latency path.

[dsv3_fused_a_gemm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1296) **(sgl-kernel)**
└─> [self.self_attn.prepare_qkv_latent()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2164)
    └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
        └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
            └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 1.10 dsv3_router_gemm **(sgl-kernel)**

Router GEMM for DeepSeek V3 MoE gating.

[dsv3_router_gemm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L330) **(sgl-kernel)**
└─> [self.gate()](../sglang/python/sglang/srt/models/deepseek_v2.py#L561)
    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 2. Activation Kernels

#### 2.1 silu_and_mul **(sgl-kernel)**

SiLU activation with gate multiplication (SwiGLU).

**Path 1: Dense MLP**

[silu_and_mul()](../sglang/python/sglang/srt/layers/activation.py#L58) **(sgl-kernel)**
└─> [self.act_fn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L272)
    └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 2: MoE (Triton runner)**

[silu_and_mul()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L195) **(sgl-kernel)**
└─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L997)
    └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
        └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
            └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 3: MoE (Fused experts Triton)**

[silu_and_mul()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L459) **(sgl-kernel)**
└─> [fused_experts()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L314)
    └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L973)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 4: Cutlass MoE**

[silu_and_mul()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L199) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
        └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
            └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 2.2 gelu_and_mul **(sgl-kernel)**

GELU activation with gate multiplication.

**Path 1: Dense MLP**

[gelu_and_mul()](../sglang/python/sglang/srt/layers/activation.py#L74) **(sgl-kernel)**
└─> [self.act_fn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L272)
    └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

**Path 2: MoE (Triton fused experts)**

[gelu_and_mul()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L475) **(sgl-kernel)**
└─> [fused_experts()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L314)
    └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L973)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 2.3 gelu_tanh_and_mul **(sgl-kernel)**

GELU-tanh activation with gate multiplication.

[gelu_tanh_and_mul()](../sglang/python/sglang/srt/layers/activation.py#L72) **(sgl-kernel)**
└─> [self.act_fn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L272)
    └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 3. Normalization Kernels

#### 3.1 rmsnorm **(sgl-kernel)**

RMSNorm without residual addition.

[rmsnorm()](../sglang/python/sglang/srt/layers/layernorm.py#L102) **(sgl-kernel)**
├─> [self.input_layernorm()](../sglang/python/sglang/srt/layers/communicator.py#L389)
│   └─> [self.layer_communicator.prepare_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2197)
│       └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│           └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.post_attention_layernorm()](../sglang/python/sglang/srt/layers/communicator.py#L438)
│   └─> [self.layer_communicator.prepare_mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2212)
│       └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│           └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.q_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1323) (NSA path)
│   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.q_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1336) (non-NSA path)
│   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.kv_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1348)
│   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.q_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1440) (alt_stream)
│   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.kv_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1442) (alt_stream)
│   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.q_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1445) (normal)
│   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.kv_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1446) (normal)
│   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.kv_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1491) (no q_lora_rank)
│   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.q_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1690)
│   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
└─> [self.kv_a_layernorm()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1712)
    └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
        └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
            └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 3.2 fused_add_rmsnorm **(sgl-kernel)**

RMSNorm with fused residual addition.

[fused_add_rmsnorm()](../sglang/python/sglang/srt/layers/layernorm.py#L100) **(sgl-kernel)**
├─> [self.input_layernorm()](../sglang/python/sglang/srt/layers/communicator.py#L389)
│   └─> [self.layer_communicator.prepare_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2197)
│       └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│           └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
└─> [self.post_attention_layernorm()](../sglang/python/sglang/srt/layers/communicator.py#L438)
    └─> [self.layer_communicator.prepare_mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2212)
        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 4. GEMM Kernels

#### 4.1 int8_scaled_mm **(sgl-kernel)**

INT8 scaled matrix multiplication for W8A8 quantization.

[int8_scaled_mm()](../sglang/python/sglang/srt/layers/quantization/w8a8_int8.py#L198) **(sgl-kernel)**
├─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L221) **(ReplicatedLinear)**
│   ├─> [self.fused_qkv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1300)
│   │   └─> [self.qkv_latent_func()](../sglang/python/sglang/srt/layers/communicator.py#L110)
│   │       └─> [fetch_qkv_latent()](../sglang/python/sglang/srt/layers/communicator.py#L176)
│   │           └─> [.fetch_qkv_latent()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1313)
│   │               └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │                   └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │                       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │                           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.fused_qkv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1687)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1343)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1489)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   └─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1696)
│       └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│           └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│               └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│                   └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                       └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L381) **(ColumnParallelLinear)**
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1340)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1486)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1693)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1324) (NSA path)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1337) (non-NSA path)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1463) (alt_stream)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1476) (normal)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1691)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1372)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   └─> [self.gate_up_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L271)
│       └─> `DeepseekV2MLP.forward()`
│           └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
└─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L1225) **(RowParallelLinear)**
    ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
    │   └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
    │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1661)
    │   └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
    │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1843)
    │   └─> [self.forward_absorb_fused_mla_rope_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1282)
    │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    └─> [self.down_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L273)
        └─> `DeepseekV2MLP.forward()`
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 4.2 fp8_blockwise_scaled_mm **(sgl-kernel)**

FP8 blockwise scaled matrix multiplication.

[fp8_blockwise_scaled_mm()](../sglang/python/sglang/srt/layers/quantization/fp8_utils.py#L333) **(sgl-kernel)**
└─> [self.w8a8_block_fp8_linear()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L449)
├─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L221) **(ReplicatedLinear)**
│   ├─> [self.fused_qkv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1300)
│   │   └─> [self.qkv_latent_func()](../sglang/python/sglang/srt/layers/communicator.py#L110)
│   │       └─> [fetch_qkv_latent()](../sglang/python/sglang/srt/layers/communicator.py#L176)
│   │           └─> [.fetch_qkv_latent()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1313)
│   │               └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │                   └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │                       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │                           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.fused_qkv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1687)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1343)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1489)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   └─> [self.kv_a_proj_with_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1696)
│       └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│           └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│               └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│                   └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                       └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L381) **(ColumnParallelLinear)**
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1340)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1486)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1693)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1324) (NSA path)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1337) (non-NSA path)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1463) (alt_stream)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1476) (normal)
│   │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.q_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1691)
│   │   └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.kv_b_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1372)
│   │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
│   │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
│   │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   └─> [self.gate_up_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L271)
│       └─> `DeepseekV2MLP.forward()`
│           └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
├─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L1225) **(RowParallelLinear)**
│   ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
│   │   └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
│   │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1661)
│   │   └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
│   │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   ├─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1843)
│   │   └─> [self.forward_absorb_fused_mla_rope_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1282)
│   │       └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
│   │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│   │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
│   └─> [self.down_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L273)
│       └─> `DeepseekV2MLP.forward()`
│           └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
│               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
│                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
└─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875) **(FusedMoE - shared_experts)**
    └─> [run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
        └─> [forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
            └─> [self.shared_experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L814)
                └─> [self._forward_shared_experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L555)
                    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 4.3 fp8_scaled_mm **(sgl-kernel)**

FP8 scaled GEMM (non-blockwise).

[fp8_scaled_mm()](../sglang/python/sglang/srt/layers/quantization/fp8_utils.py#L909) **(sgl-kernel)**
└─> [apply_fp8_linear()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L458)
    └─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L449)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L381)
            └─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
                └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
                    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 4.4 cutlass_scaled_fp4_mm **(sgl-kernel)**

CUTLASS FP4 GEMM for quantized linear layers (ModelOpt FP4 path).

[cutlass_scaled_fp4_mm()](../sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L135) **(sgl-kernel)**
└─> [fp4_gemm()](../sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L119)
    └─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L703)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/linear.py#L381)
            └─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
                └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
                    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 4.5 qserve_w4a8_per_chn_gemm **(sgl-kernel)**

QServe W4A8 per-channel GEMM.

[qserve_w4a8_per_chn_gemm()](../sglang/python/sglang/srt/layers/quantization/qoq.py#L230) **(sgl-kernel)**
└─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/qoq.py#L220)
    └─> [self.linear()](../sglang/python/sglang/srt/layers/linear.py#L380)
        └─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
            └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 4.6 qserve_w4a8_per_group_gemm **(sgl-kernel)**

QServe W4A8 per-group GEMM.

[qserve_w4a8_per_group_gemm()](../sglang/python/sglang/srt/layers/quantization/qoq.py#L235) **(sgl-kernel)**
└─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/qoq.py#L220)
    └─> [self.linear()](../sglang/python/sglang/srt/layers/linear.py#L380)
        └─> [self.o_proj()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
            └─> [self.forward_normal_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1274)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 5. MoE Kernels

#### 5.1 fp8_blockwise_scaled_grouped_mm **(sgl-kernel)**

FP8 grouped GEMM for MoE experts.

[fp8_blockwise_scaled_grouped_mm()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L177) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
        └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
            └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.2 fused_moe_kernel **(triton)**

Triton fused MoE GEMM kernel.

[fused_moe_kernel[grid]()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L770) **(triton)**
└─> [invoke_fused_moe_kernel()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L421)
    └─> [fused_experts_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L67)
        └─> [inplace_fused_experts()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L189)
            └─> [fused_experts()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L314)
                └─> [self.fused_func()](../sglang/python/sglang/srt/layers/moe/moe_runner/runner.py#L76)
                    └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L997)
                        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
                            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.3 moe_align_block_size **(sgl-kernel)**

Align MoE block sizes for efficient processing.

[sgl_moe_align_block_size()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py#L74) **(sgl-kernel)**
└─> [moe_align_block_size()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L417)
    └─> [fused_experts_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L67)
        └─> [inplace_fused_experts()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L189)
            └─> [fused_experts()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L314)
                └─> [self.fused_func()](../sglang/python/sglang/srt/layers/moe/moe_runner/runner.py#L76)
                    └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L997)
                        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
                            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.4 trtllm_bf16_moe **(flashinfer)**

BF16 MoE kernel for FlashInferFusedMoE (unquantized).

[trtllm_bf16_moe()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1045) **(flashinfer)**
└─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1007)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.5 trtllm_fp8_block_scale_moe **(flashinfer)**

FP8 block scale MoE kernel for FlashInferFusedMoE.

[trtllm_fp8_block_scale_moe()](../sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py#L168) **(flashinfer)**
└─> [self.fused_func()](../sglang/python/sglang/srt/layers/moe/moe_runner/runner.py#L76)
    └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L997)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.6 trtllm_fp4_block_scale_moe **(flashinfer)**

FP4 block scale MoE kernel for FlashInferFP4MoE.

[trtllm_fp4_block_scale_moe()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1188) **(flashinfer)**
└─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1138)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.7 prepare_moe_input **(sgl-kernel)**

Prepare MoE input (compute a_map, c_map, expert_offsets) for Cutlass MoE.

[prepare_moe_input()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L140) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.8 shuffle_rows **(sgl-kernel)**

Shuffle rows for Cutlass MoE expert reordering.

[shuffle_rows()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L153) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.9 es_fp8_blockwise_scaled_grouped_mm **(sgl-kernel)**

Expert-specialized FP8 block-scaled grouped GEMM for Cutlass MoE.

[es_fp8_blockwise_scaled_grouped_mm()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L163) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.10 apply_shuffle_mul_sum **(sgl-kernel)**

Apply shuffle, multiply by weights, and sum for Cutlass MoE output.

[apply_shuffle_mul_sum()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L242) **(sgl-kernel)**
└─> [cutlass_fused_experts_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L865)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.11 scaled_fp4_experts_quant **(sgl-kernel)**

FP4 quantization for Cutlass FP4 MoE.

[scaled_fp4_experts_quant()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L353) **(sgl-kernel)**
└─> [cutlass_moe_fp4()](../sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L1721)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.12 cutlass_fp4_group_mm **(sgl-kernel)**

FP4 grouped GEMM for Cutlass FP4 MoE.

[cutlass_fp4_group_mm()](../sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L361) **(sgl-kernel)**
└─> [cutlass_moe_fp4()](../sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L1721)
    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.13 moe_sum_reduce **(sgl-kernel)**

Sum reduction for MoE output accumulation.

[moe_sum_reduce()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L551) **(sgl-kernel)**
└─> [fused_experts_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L67)
    └─> [inplace_fused_experts()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L189)
        └─> [fused_experts()](../sglang/python/sglang/srt/layers/moe/moe_runner/triton.py#L314)
            └─> [self.fused_func()](../sglang/python/sglang/srt/layers/moe/moe_runner/runner.py#L76)
                └─> [self.runner.run()](../sglang/python/sglang/srt/layers/quantization/fp8.py#L997)
                    └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
                        └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                            └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                                └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                                    └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                                        └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.14 cutlass_w4a8_moe_mm **(sgl-kernel)**

CUTLASS W4A8 MoE GEMM.

[cutlass_w4a8_moe_mm()](../sglang/python/sglang/srt/layers/moe/cutlass_w4a8_moe.py#L155) **(sgl-kernel)**
└─> [cutlass_w4a8_moe()](../sglang/python/sglang/srt/layers/quantization/w4afp8.py#L305)
    └─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/w4afp8.py#L281)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 5.15 get_cutlass_w4a8_moe_mm_data **(sgl-kernel)**

Prepare data for CUTLASS W4A8 MoE GEMM.

[get_cutlass_w4a8_moe_mm_data()](../sglang/python/sglang/srt/layers/moe/cutlass_w4a8_moe.py#L140) **(sgl-kernel)**
└─> [cutlass_w4a8_moe()](../sglang/python/sglang/srt/layers/quantization/w4afp8.py#L305)
    └─> [self.apply()](../sglang/python/sglang/srt/layers/quantization/w4afp8.py#L281)
        └─> [self.quant_method.apply()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L875)
            └─> [self.run_moe_core()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L854)
                └─> [self.forward_impl()](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L844)
                    └─> [self.experts()](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
                        └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
                            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 6. MoE Gating Kernels

#### 6.1 topk_softmax **(sgl-kernel)**

TopK with softmax for MoE gating.

[topk_softmax()](../sglang/python/sglang/srt/layers/moe/topk.py#L404) **(sgl-kernel)**
└─> [fused_topk()](../sglang/python/sglang/srt/layers/moe/topk.py#L802)
    └─> [select_experts()](../sglang/python/sglang/srt/layers/moe/topk.py#L284)
    ├─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L562)
    │   └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
    │       └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    └─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L594)
        └─> [self.forward_normal()](../sglang/python/sglang/srt/models/deepseek_v2.py#L536)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 6.2 topk_sigmoid **(sgl-kernel)**

TopK with sigmoid for MoE gating.

[topk_sigmoid()](../sglang/python/sglang/srt/layers/moe/topk.py#L411) **(sgl-kernel)**
└─> [fused_topk()](../sglang/python/sglang/srt/layers/moe/topk.py#L802)
    └─> [select_experts()](../sglang/python/sglang/srt/layers/moe/topk.py#L284)
    ├─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L562)
    │   └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
    │       └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    └─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L594)
        └─> [self.forward_normal()](../sglang/python/sglang/srt/models/deepseek_v2.py#L536)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 6.3 moe_fused_gate **(sgl-kernel)**

Fused MoE gating kernel (used for biased grouped topk with DeepSeek V3 style routing).

[moe_fused_gate()](../sglang/python/sglang/srt/layers/moe/topk.py#L643) **(sgl-kernel)**
└─> [biased_grouped_topk()](../sglang/python/sglang/srt/layers/moe/topk.py#L771)
    └─> [select_experts()](../sglang/python/sglang/srt/layers/moe/topk.py#L284)
    ├─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L562)
    │   └─> [self.forward_normal_dual_stream()](../sglang/python/sglang/srt/models/deepseek_v2.py#L529)
    │       └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
    │           └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │               └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    └─> [self.topk()](../sglang/python/sglang/srt/models/deepseek_v2.py#L594)
        └─> [self.forward_normal()](../sglang/python/sglang/srt/models/deepseek_v2.py#L536)
            └─> [self.mlp()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2230)
                └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                    └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 7. RoPE Kernels

#### 7.1 apply_rope_with_cos_sin_cache_inplace **(sgl-kernel)**

Rotary position embedding kernel.

[apply_rope_with_cos_sin_cache_inplace()](../sglang/python/sglang/srt/layers/rotary_embedding.py#L242) **(sgl-kernel)**
└─> [self._forward_method()](../sglang/python/sglang/srt/layers/utils/multi_platform.py#L57)
    ├─> [self.rotary_emb()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1354)
    │   └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
    │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
    │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
    │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    ├─> [self.rotary_emb()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1533)
    │   └─> [self.forward_absorb_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1255)
    │       └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
    │           └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
    │               └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
    │                   └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)
    └─> [self.rotary_emb()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1718)
        └─> [self.forward_absorb_fused_mla_rope_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1259)
            └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                    └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                        └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 7.2 mla_rope_quantize_fp8 **(flashinfer)**

MLA-specific RoPE with FP8 quantization (TRTLLM path).

[mla_rope_quantize_fp8()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L726) **(flashinfer)**
└─> [self.quantize_and_rope_for_fp8()](../sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L849)
    └─> [self.forward_decode()](../sglang/python/sglang/srt/layers/attention/base_attn_backend.py#L90)
        └─> [attn_backend.forward()](../sglang/python/sglang/srt/layers/radix_attention.py#L124)
            └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
                └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                    └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                        └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                            └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                                └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

#### 7.3 rotary_embedding **(sgl-kernel)**

Fallback rotary position embedding kernel (used when apply_rope_with_cos_sin_cache_inplace is not available).

[self.fallback_rotary_embedding()](../sglang/python/sglang/srt/layers/rotary_embedding.py#L260) **(sgl-kernel)**
└─> [self._forward_method()](../sglang/python/sglang/srt/layers/utils/multi_platform.py#L57)
    └─> [self.rotary_emb()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1354)
        └─> [self.forward_normal_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1243)
            └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                    └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                        └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

### 8. Quantization Kernels

These kernels are called during FP8/INT8 quantization for linear layers.

#### 8.1 sgl_per_token_quant_fp8 **(sgl-kernel)**

Per-token FP8 quantization.

[sgl_per_token_quant_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#L544) **(sgl-kernel)**
└─> **(called during FP8 linear layer quantization)**

#### 8.2 sgl_per_token_group_quant_fp8 **(sgl-kernel)**

Per-token group FP8 quantization.

[sgl_per_token_group_quant_fp8()](../sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#L480) **(sgl-kernel)**
└─> **(called during FP8 linear layer quantization)**

#### 8.3 sgl_per_token_group_quant_8bit **(sgl-kernel)**

Per-token group 8-bit quantization.

[sgl_per_token_group_quant_8bit()](../sglang/python/sglang/srt/layers/quantization/int8_kernel.py#L218) **(sgl-kernel)**
└─> **(called during INT8 linear layer quantization)**

#### 8.4 sgl_per_token_group_quant_int8 **(sgl-kernel)**

Per-token group INT8 quantization.

[sgl_per_token_group_quant_int8()](../sglang/python/sglang/srt/layers/quantization/int8_kernel.py#L223) **(sgl-kernel)**
└─> **(called during INT8 linear layer quantization)**

---

### 9. NSA Kernels

NSA (Native Sparse Attention) kernels for DeepSeek models with sparse attention.

#### 9.1 hadamard_transform **(sgl-kernel)**

Hadamard transform for NSA indexer.

[hadamard_transform()](../sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py#L105) **(sgl-kernel)**
└─> **(NSA indexer forward)**

#### 9.2 fast_topk_v2 **(sgl-kernel)**

Fast TopK v2 for NSA sparse attention.

[fast_topk_v2()](../sglang/python/sglang/srt/layers/attention/nsa_backend.py#L211) **(sgl-kernel)**
└─> **(NSA decode forward)**

#### 9.3 fast_topk_transform_fused **(sgl-kernel)**

Fused fast TopK transform for NSA.

[fast_topk_transform_fused()](../sglang/python/sglang/srt/layers/attention/nsa_backend.py#L214) **(sgl-kernel)**
└─> **(NSA decode forward)**

#### 9.4 fast_topk_transform_ragged_fused **(sgl-kernel)**

Fused fast TopK transform for ragged NSA.

[fast_topk_transform_ragged_fused()](../sglang/python/sglang/srt/layers/attention/nsa_backend.py#L223) **(sgl-kernel)**
└─> **(NSA extend forward)**

---

### 10. Sampling Kernels

These kernels are called during token sampling after model forward.

#### 10.1 min_p_sampling_from_probs **(sgl-kernel)**

Min-P sampling from probability distribution.

[min_p_sampling_from_probs()](../sglang/python/sglang/srt/layers/sampler.py#L133) **(sgl-kernel)**
└─> **(post-forward sampling)**

#### 10.2 top_k_renorm_prob **(sgl-kernel)**

Top-K probability renormalization.

[top_k_renorm_prob()](../sglang/python/sglang/srt/layers/sampler.py#L131) **(sgl-kernel)**
└─> **(post-forward sampling)**

#### 10.3 top_p_renorm_prob **(sgl-kernel)**

Top-P probability renormalization.

[top_p_renorm_prob()](../sglang/python/sglang/srt/layers/sampler.py#L132) **(sgl-kernel)**
└─> **(post-forward sampling)**

#### 10.4 top_k_top_p_sampling_from_probs **(sgl-kernel)**

Combined Top-K/Top-P sampling from probability distribution.

[top_k_top_p_sampling_from_probs()](../sglang/python/sglang/srt/layers/sampler.py#L137) **(sgl-kernel)**
└─> **(post-forward sampling)**

---

### 11. Speculative Decoding Kernels

These kernels are used by EAGLE speculative decoding.

#### 11.1 fast_topk **(sgl-kernel)**

Fast TopK for speculative decoding.

[fast_topk()](../sglang/python/sglang/srt/speculative/eagle_worker.py#L649) **(sgl-kernel)**
└─> **(EAGLE speculative decoding)**

#### 11.2 build_tree_kernel_efficient **(sgl-kernel)**

Efficient tree building kernel for speculative decoding.

[build_tree_kernel_efficient()](../sglang/python/sglang/srt/speculative/eagle_worker.py#L567) **(sgl-kernel)**
└─> **(EAGLE speculative decoding)**

#### 11.3 verify_tree_greedy **(sgl-kernel)**

Greedy tree verification for speculative decoding.

[verify_tree_greedy()](../sglang/python/sglang/srt/speculative/eagle_utils.py#L157) **(sgl-kernel)**
└─> **(EAGLE speculative decoding verification)**

#### 11.4 tree_speculative_sampling_target_only **(sgl-kernel)**

Tree speculative sampling kernel (target-only).

[tree_speculative_sampling_target_only()](../sglang/python/sglang/srt/speculative/eagle_info.py#L350) **(sgl-kernel)**
└─> **(EAGLE speculative decoding verification)**

---

### 12. Utility Kernels

#### 12.1 weak_ref_tensor **(sgl-kernel)**

Weak reference tensor for memory management.

[weak_ref_tensor()](../sglang/python/sglang/srt/compilation/weak_ref_tensor.py#L21) **(sgl-kernel)**
└─> [convert_to_weak_ref()](../sglang/python/sglang/srt/compilation/weak_ref_tensor.py#L17)
    └─> **(compilation utility)**

---

### 13. Attention Utility Kernels

#### 13.1 cutlass_mla_get_workspace_size **(sgl-kernel)**

Get workspace size for CUTLASS MLA decode kernel.

[cutlass_mla_get_workspace_size()](../sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py#L107) **(sgl-kernel)**
└─> [init_forward_metadata()](../sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py#L100)
    └─> [attn_backend.init_forward_metadata()](../sglang/python/sglang/srt/layers/radix_attention.py#L90)
        └─> [self.attn_mqa()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
            └─> [self.forward_absorb_core()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1280)
                └─> [self.forward_prepare()](../sglang/python/sglang/srt/models/deepseek_v2.py#L1201)
                    └─> [self.self_attn()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2204)
                        └─> [layer()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2492)
                            └─> [self.model()](../sglang/python/sglang/srt/models/deepseek_v2.py#L2670)

---

## Kernel Source Breakdown

### sgl-kernel (CUDA)

Located at: `sglang/sgl-kernel/`

| Kernel | Module | File |
|--------|--------|------|
| `cutlass_mla_decode` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `cutlass_mla_get_workspace_size` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `merge_state` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `merge_state_v2` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `concat_mla_k` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `concat_mla_absorb_q` | attention | ../sglang/sgl-kernel/python/sgl_kernel/attention.py |
| `silu_and_mul` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `gelu_and_mul` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `gelu_tanh_and_mul` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `rmsnorm` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `fused_add_rmsnorm` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `apply_rope_with_cos_sin_cache_inplace` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `rotary_embedding` | elementwise | ../sglang/sgl-kernel/python/sgl_kernel/elementwise.py |
| `int8_scaled_mm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `fp8_scaled_mm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `fp8_blockwise_scaled_mm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `cutlass_scaled_fp4_mm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `bmm_fp8` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `dsv3_fused_a_gemm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `dsv3_router_gemm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `qserve_w4a8_per_chn_gemm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `qserve_w4a8_per_group_gemm` | gemm | ../sglang/sgl-kernel/python/sgl_kernel/gemm.py |
| `fp8_blockwise_scaled_grouped_mm` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `es_fp8_blockwise_scaled_grouped_mm` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `prepare_moe_input` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `shuffle_rows` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `apply_shuffle_mul_sum` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `moe_sum_reduce` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `scaled_fp4_experts_quant` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `cutlass_fp4_group_mm` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `cutlass_w4a8_moe_mm` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `get_cutlass_w4a8_moe_mm_data` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `topk_softmax` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `topk_sigmoid` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `moe_fused_gate` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `moe_align_block_size` | moe | ../sglang/sgl-kernel/python/sgl_kernel/moe.py |
| `sgl_per_token_quant_fp8` | quantization | ../sglang/sgl-kernel/python/sgl_kernel/quantization.py |
| `sgl_per_token_group_quant_fp8` | quantization | ../sglang/sgl-kernel/python/sgl_kernel/quantization.py |
| `sgl_per_token_group_quant_8bit` | quantization | ../sglang/sgl-kernel/python/sgl_kernel/quantization.py |
| `sgl_per_token_group_quant_int8` | quantization | ../sglang/sgl-kernel/python/sgl_kernel/quantization.py |
| `hadamard_transform` | nsa | ../sglang/sgl-kernel/python/sgl_kernel/nsa.py |
| `fast_topk` | speculative | ../sglang/sgl-kernel/python/sgl_kernel/speculative.py |
| `fast_topk_v2` | nsa | ../sglang/sgl-kernel/python/sgl_kernel/nsa.py |
| `fast_topk_transform_fused` | nsa | ../sglang/sgl-kernel/python/sgl_kernel/nsa.py |
| `fast_topk_transform_ragged_fused` | nsa | ../sglang/sgl-kernel/python/sgl_kernel/nsa.py |
| `min_p_sampling_from_probs` | sampling | ../sglang/sgl-kernel/python/sgl_kernel/sampling.py |
| `top_k_renorm_prob` | sampling | ../sglang/sgl-kernel/python/sgl_kernel/sampling.py |
| `top_p_renorm_prob` | sampling | ../sglang/sgl-kernel/python/sgl_kernel/sampling.py |
| `top_k_top_p_sampling_from_probs` | sampling | ../sglang/sgl-kernel/python/sgl_kernel/sampling.py |
| `build_tree_kernel_efficient` | speculative | ../sglang/sgl-kernel/python/sgl_kernel/speculative.py |
| `verify_tree_greedy` | speculative | ../sglang/sgl-kernel/python/sgl_kernel/speculative.py |
| `tree_speculative_sampling_target_only` | speculative | ../sglang/sgl-kernel/python/sgl_kernel/speculative.py |
| `weak_ref_tensor` | utility | ../sglang/sgl-kernel/python/sgl_kernel/utility.py |

### flashinfer

Located at: `flashinfer/` (cloned from https://github.com/lpc0220/flashinfer)

| Kernel | Module | Description |
|--------|--------|-------------|
| `trtllm_batch_decode_with_kv_cache_mla` | decode | MLA decode kernel |
| `trtllm_ragged_attention_deepseek` | prefill | DeepSeek prefill attention |
| `mla_rope_quantize_fp8` | rope | MLA RoPE with FP8 |
| `trtllm_bf16_moe` | fused_moe | BF16 MoE kernel |
| `trtllm_fp8_block_scale_moe` | fused_moe | FP8 block scale MoE kernel |
| `trtllm_fp4_block_scale_moe` | fused_moe | FP4 block scale MoE kernel |

### triton (JIT)

Located in: `sglang/python/sglang/srt/layers/`

| Kernel | File | Description |
|--------|------|-------------|
| `fused_moe_kernel` | [fused_moe_triton_kernels.py](../sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py) | MoE GEMM kernel |

---

## DeepSeek Model Entry Points

All kernel traces ultimately lead to these DeepSeek model files:

| Model | File | Main Entry |
|-------|------|------------|
| DeepSeek V2/V3/R1 | [deepseek_v2.py](../sglang/python/sglang/srt/models/deepseek_v2.py) | `DeepseekV2ForCausalLM.forward()` |
| DeepSeek V1 | [deepseek.py](../sglang/python/sglang/srt/models/deepseek.py) | `DeepseekForCausalLM.forward()` |
| DeepSeek NextN | [deepseek_nextn.py](../sglang/python/sglang/srt/models/deepseek_nextn.py) | `DeepSeekNextN.forward()` |

---

## Notes

- All paths use `../` prefix for VSCode clickable links (output file is in `results/` directory)
- Call chains show **call sites** (where functions are called), not function definitions
- The format follows GDB's `bt` (backtrace) style
- Depth is indicated by the number of dashes in `└─>`, `└─>`, `└─>`, etc.
- **Branching trees** use `├─>` for intermediate branches and `└─>` for final branches
- Multiple paths may exist for the same kernel depending on:
  - Quantization method (FP8, INT8, FP16)
  - Backend selection (TRTLLM, FlashInfer, Triton)
  - Layer type (dense MLP vs MoE)
  - Attention path (forward_normal_*, forward_absorb_*, forward_absorb_fused_mla_rope_*)


---


---
