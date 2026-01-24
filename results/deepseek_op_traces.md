# DeepSeek Op Traces

This document maps DeepSeek model ops (logical operations) to their kernel implementations, organized in execution order.

**Generated:** 2026-01-20

**Input:** `results/deepseek_kernel_traces.md` - Kernel call chains (60 kernels)

---

## Execution Graph

The execution graph shows the op hierarchy in DeepSeek V2/V3/R1 models. Each op lists ALL call sites (one link per line).

self.model()
└─> layer()
    ├─> self.layer_communicator.prepare_attn()
    │   └─> self.input_layernorm()
    │       - [L389](../sglang/python/sglang/srt/layers/communicator.py#L389) *(no residual)*
    │       - [L391](../sglang/python/sglang/srt/layers/communicator.py#L391) *(with residual)*
    │
    ├─> self.self_attn()
    │   ├─> self.forward_prepare()
    │   │   │
    │   │   ├─> self.forward_normal_prepare() **(MHA path)**
    │   │   │   │
    │   │   │   ├─> self.fused_qkv_a_proj_with_mqa()
    │   │   │   │   - [L1300](../sglang/python/sglang/srt/models/deepseek_v2.py#L1300)
    │   │   │   │
    │   │   │   ├─> self.q_a_layernorm()
    │   │   │   │   - [L1323](../sglang/python/sglang/srt/models/deepseek_v2.py#L1323)
    │   │   │   │   - [L1336](../sglang/python/sglang/srt/models/deepseek_v2.py#L1336)
    │   │   │   │
    │   │   │   ├─> self.q_b_proj()
    │   │   │   │   - [L1324](../sglang/python/sglang/srt/models/deepseek_v2.py#L1324)
    │   │   │   │   - [L1337](../sglang/python/sglang/srt/models/deepseek_v2.py#L1337)
    │   │   │   │
    │   │   │   ├─> self.q_proj()
    │   │   │   │   - [L1340](../sglang/python/sglang/srt/models/deepseek_v2.py#L1340)
    │   │   │   │
    │   │   │   ├─> self.kv_a_proj_with_mqa()
    │   │   │   │   - [L1343](../sglang/python/sglang/srt/models/deepseek_v2.py#L1343)
    │   │   │   │
    │   │   │   ├─> self.kv_a_layernorm()
    │   │   │   │   - [L1348](../sglang/python/sglang/srt/models/deepseek_v2.py#L1348)
    │   │   │   │
    │   │   │   ├─> self.rotary_emb()
    │   │   │   │   - [L1354](../sglang/python/sglang/srt/models/deepseek_v2.py#L1354)
    │   │   │   │
    │   │   │   ├─> self.kv_b_proj()
    │   │   │   │   - [L1372](../sglang/python/sglang/srt/models/deepseek_v2.py#L1372)
    │   │   │   │
    │   │   │   └─> self._concat_and_cast_mha_k()
    │   │   │       - [L1377](../sglang/python/sglang/srt/models/deepseek_v2.py#L1377)
    │   │   │
    │   │   ├─> self.forward_absorb_prepare() **(MLA path)**
    │   │   │   │
    │   │   │   ├─> self.fused_qkv_a_proj_with_mqa() *(via prepare_qkv_latent, when q_lora_rank)*
    │   │   │   │   - [L1296](../sglang/python/sglang/srt/models/deepseek_v2.py#L1296) *(dsv3_fused_a_gemm path)*
    │   │   │   │   - [L1300](../sglang/python/sglang/srt/models/deepseek_v2.py#L1300)
    │   │   │   │
    │   │   │   ├─> self.q_a_layernorm()
    │   │   │   │   - [L1440](../sglang/python/sglang/srt/models/deepseek_v2.py#L1440)
    │   │   │   │   - [L1445](../sglang/python/sglang/srt/models/deepseek_v2.py#L1445)
    │   │   │   │
    │   │   │   ├─> self.kv_a_layernorm()
    │   │   │   │   - [L1442](../sglang/python/sglang/srt/models/deepseek_v2.py#L1442)
    │   │   │   │   - [L1446](../sglang/python/sglang/srt/models/deepseek_v2.py#L1446)
    │   │   │   │   - [L1491](../sglang/python/sglang/srt/models/deepseek_v2.py#L1491)
    │   │   │   │
    │   │   │   ├─> self.q_b_proj()
    │   │   │   │   - [L1463](../sglang/python/sglang/srt/models/deepseek_v2.py#L1463)
    │   │   │   │   - [L1476](../sglang/python/sglang/srt/models/deepseek_v2.py#L1476)
    │   │   │   │
    │   │   │   ├─> self.q_proj()
    │   │   │   │   - [L1486](../sglang/python/sglang/srt/models/deepseek_v2.py#L1486)
    │   │   │   │
    │   │   │   ├─> self.kv_a_proj_with_mqa()
    │   │   │   │   - [L1489](../sglang/python/sglang/srt/models/deepseek_v2.py#L1489)
    │   │   │   │
    │   │   │   ├─> bmm_fp8() *(q_nope * w_kc)*
    │   │   │   │   - [L1521](../sglang/python/sglang/srt/models/deepseek_v2.py#L1521)
    │   │   │   │
    │   │   │   └─> self.rotary_emb()
    │   │   │       - [L1533](../sglang/python/sglang/srt/models/deepseek_v2.py#L1533)
    │   │   │
    │   │   └─> self.forward_absorb_fused_mla_rope_prepare() **(MLA_FUSED_ROPE path)**
    │   │       │
    │   │       ├─> self.fused_qkv_a_proj_with_mqa()
    │   │       │   - [L1687](../sglang/python/sglang/srt/models/deepseek_v2.py#L1687)
    │   │       │
    │   │       ├─> self.q_a_layernorm()
    │   │       │   - [L1690](../sglang/python/sglang/srt/models/deepseek_v2.py#L1690)
    │   │       │
    │   │       ├─> self.q_b_proj()
    │   │       │   - [L1691](../sglang/python/sglang/srt/models/deepseek_v2.py#L1691)
    │   │       │
    │   │       ├─> self.q_proj()
    │   │       │   - [L1693](../sglang/python/sglang/srt/models/deepseek_v2.py#L1693)
    │   │       │
    │   │       ├─> self.kv_a_proj_with_mqa()
    │   │       │   - [L1696](../sglang/python/sglang/srt/models/deepseek_v2.py#L1696)
    │   │       │
    │   │       ├─> bmm_fp8() *(q_nope * w_kc)*
    │   │       │   - [L1705](../sglang/python/sglang/srt/models/deepseek_v2.py#L1705)
    │   │       │
    │   │       ├─> self.kv_a_layernorm()
    │   │       │   - [L1712](../sglang/python/sglang/srt/models/deepseek_v2.py#L1712)
    │   │       │
    │   │       └─> self.rotary_emb()
    │   │           - [L1718](../sglang/python/sglang/srt/models/deepseek_v2.py#L1718)
    │   │
    │   └─> self.forward_core()
    │       │
    │       ├─> self.forward_normal_core() **(MHA path)**
    │       │   │
    │       │   ├─> self.attn_mha()
    │       │   │   - [L1381](../sglang/python/sglang/srt/models/deepseek_v2.py#L1381)
    │       │   │
    │       │   └─> self.o_proj()
    │       │       - [L1383](../sglang/python/sglang/srt/models/deepseek_v2.py#L1383)
    │       │
    │       ├─> self.forward_normal_chunked_kv_core() **(MHA chunked path)**
    │       │   │
    │       │   ├─> self.attn_mha()
    │       │   │   - [L1923](../sglang/python/sglang/srt/models/deepseek_v2.py#L1923)
    │       │   │
    │       │   ├─> self._chunked_prefix_attn_mha()
    │       │   │   │
    │       │   │   ├─> self.kv_b_proj()
    │       │   │   │   - [L1864](../sglang/python/sglang/srt/models/deepseek_v2.py#L1864)
    │       │   │   │
    │       │   │   ├─> self.attn_mha()
    │       │   │   │   - [L1883](../sglang/python/sglang/srt/models/deepseek_v2.py#L1883)
    │       │   │   │
    │       │   │   └─> merge_state_v2()
    │       │   │       - [L1886](../sglang/python/sglang/srt/models/deepseek_v2.py#L1886)
    │       │   │
    │       │   └─> self.o_proj()
    │       │       - [L1937](../sglang/python/sglang/srt/models/deepseek_v2.py#L1937)
    │       │
    │       ├─> self.forward_absorb_core() **(MLA path)**
    │       │   │
    │       │   ├─> self.attn_mqa()
    │       │   │   - [L1576](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
    │       │   │   - [L1594](../sglang/python/sglang/srt/models/deepseek_v2.py#L1594) *(with context parallel)*
    │       │   │
    │       │   ├─> bmm_fp8() *(attn_output * w_vc)*
    │       │   │   - [L1632](../sglang/python/sglang/srt/models/deepseek_v2.py#L1632)
    │       │   │
    │       │   └─> self.o_proj()
    │       │       - [L1661](../sglang/python/sglang/srt/models/deepseek_v2.py#L1661)
    │       │
    │       └─> self.forward_absorb_fused_mla_rope_core() **(MLA_FUSED_ROPE path)**
    │           │
    │           ├─> self.attn_mqa()
    │           │   - [L1576](../sglang/python/sglang/srt/models/deepseek_v2.py#L1576)
    │           │   - [L1594](../sglang/python/sglang/srt/models/deepseek_v2.py#L1594) *(with context parallel)*
    │           │
    │           ├─> bmm_fp8() *(attn_output * w_vc)*
    │           │   - [L1833](../sglang/python/sglang/srt/models/deepseek_v2.py#L1833)
    │           │
    │           └─> self.o_proj()
    │               - [L1843](../sglang/python/sglang/srt/models/deepseek_v2.py#L1843)
    │
    ├─> self.layer_communicator.prepare_mlp()
    │   └─> self.post_attention_layernorm()
    │       - [L438](../sglang/python/sglang/srt/layers/communicator.py#L438) *(passed to communicate fn)*
    │
    └─> self.mlp()
        │
        ├─> **Dense MLP - DeepseekV2MLP**
        │   │
        │   ├─> self.gate_up_proj()
        │   │   - [L271](../sglang/python/sglang/srt/models/deepseek_v2.py#L271)
        │   │
        │   ├─> self.act_fn()
        │   │   - [L272](../sglang/python/sglang/srt/models/deepseek_v2.py#L272)
        │   │
        │   └─> self.down_proj()
        │       - [L273](../sglang/python/sglang/srt/models/deepseek_v2.py#L273)
        │
        └─> **MoE MLP - DeepseekV2MoE**
            │
            ├─> self.forward_normal_dual_stream()
            │   │
            │   ├─> self._forward_shared_experts()
            │   │   └─> self.shared_experts()
            │   │       - [L555](../sglang/python/sglang/srt/models/deepseek_v2.py#L555) *(via L814)*
            │   │
            │   ├─> self.gate()
            │   │   - [L561](../sglang/python/sglang/srt/models/deepseek_v2.py#L561)
            │   │
            │   ├─> self.topk()
            │   │   - [L562](../sglang/python/sglang/srt/models/deepseek_v2.py#L562)
            │   │
            │   └─> self.experts()
            │       - [L563](../sglang/python/sglang/srt/models/deepseek_v2.py#L563)
            │
            ├─> self.forward_normal()
            │   │
            │   ├─> self._forward_shared_experts()
            │   │   └─> self.shared_experts()
            │   │       - [L589](../sglang/python/sglang/srt/models/deepseek_v2.py#L589) *(via L814)*
            │   │       - [L609](../sglang/python/sglang/srt/models/deepseek_v2.py#L609) *(via L814, hook)*
            │   │       - [L765](../sglang/python/sglang/srt/models/deepseek_v2.py#L765) *(via L814, hook)*
            │   │
            │   ├─> self.gate()
            │   │   - [L593](../sglang/python/sglang/srt/models/deepseek_v2.py#L593)
            │   │
            │   ├─> self.topk()
            │   │   - [L594](../sglang/python/sglang/srt/models/deepseek_v2.py#L594)
            │   │
            │   └─> self.experts()
            │       - [L629](../sglang/python/sglang/srt/models/deepseek_v2.py#L629)
            │
            └─> self.forward_deepep()
                │
                ├─> self.gate()
                │   - [L663](../sglang/python/sglang/srt/models/deepseek_v2.py#L663)
                │
                ├─> self._forward_shared_experts()
                │   └─> self.shared_experts()
                │       - [L668](../sglang/python/sglang/srt/models/deepseek_v2.py#L668) *(via L814)*
                │       - [L672](../sglang/python/sglang/srt/models/deepseek_v2.py#L672) *(via L814)*
                │       - [L689](../sglang/python/sglang/srt/models/deepseek_v2.py#L689) *(via L814, hook)*
                │       - [L765](../sglang/python/sglang/srt/models/deepseek_v2.py#L765) *(via L814, hook)*
                │
                ├─> self.topk()
                │   - [L673](../sglang/python/sglang/srt/models/deepseek_v2.py#L673)
                │
                └─> self.experts()
                    - [L786](../sglang/python/sglang/srt/models/deepseek_v2.py#L786)

---

## Direct Model Kernels

These kernels are called directly in model code (not through layers):

| Kernel | Call Sites |
|--------|------------|
| `dsv3_fused_a_gemm()` | [L1296](../sglang/python/sglang/srt/models/deepseek_v2.py#L1296) |
| `dsv3_router_gemm()` | [L330](../sglang/python/sglang/srt/models/deepseek_v2.py#L330) |
| `bmm_fp8()` | [L1521](../sglang/python/sglang/srt/models/deepseek_v2.py#L1521), [L1632](../sglang/python/sglang/srt/models/deepseek_v2.py#L1632), [L1705](../sglang/python/sglang/srt/models/deepseek_v2.py#L1705), [L1833](../sglang/python/sglang/srt/models/deepseek_v2.py#L1833) |
| `merge_state_v2()` | [L1886](../sglang/python/sglang/srt/models/deepseek_v2.py#L1886) |
| `concat_mla_k()` | [L2046](../sglang/python/sglang/srt/models/deepseek_v2.py#L2046) |

---

## Op Summary (Model Forward Pass)

These ops are part of the DeepSeek model's forward pass.

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `self.input_layernorm()` | 2 | rmsnorm, fused_add_rmsnorm |
| `self.fused_qkv_a_proj_with_mqa()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm, dsv3_fused_a_gemm |
| `self.q_a_layernorm()` | 1 | rmsnorm |
| `self.kv_a_layernorm()` | 1 | rmsnorm |
| `self.q_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.q_b_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.kv_a_proj_with_mqa()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.kv_b_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.rotary_emb()` | 3 | apply_rope_with_cos_sin_cache_inplace, mla_rope_quantize_fp8, rotary_embedding |
| `bmm_fp8()` | 1 | bmm_fp8 |
| `self._concat_and_cast_mha_k()` | 1 | concat_mla_k |
| `self.attn_mha()` | 2 | merge_state, merge_state_v2 |
| `self.attn_mqa()` | 5 | cutlass_mla_decode, cutlass_mla_get_workspace_size, trtllm_batch_decode_with_kv_cache_mla, trtllm_ragged_attention_deepseek, concat_mla_absorb_q |
| `self.o_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.post_attention_layernorm()` | 2 | rmsnorm, fused_add_rmsnorm |
| `self.gate_up_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.act_fn()` | 3 | silu_and_mul, gelu_and_mul, gelu_tanh_and_mul |
| `self.down_proj()` | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| `self.gate()` | 1 | dsv3_router_gemm |
| `self.topk()` | 3 | topk_softmax, topk_sigmoid, moe_fused_gate |
| `self.experts()` | 13 | fused_moe_kernel, fp8_blockwise_scaled_grouped_mm, es_fp8_blockwise_scaled_grouped_mm, prepare_moe_input, shuffle_rows, apply_shuffle_mul_sum, moe_align_block_size, moe_sum_reduce, scaled_fp4_experts_quant, cutlass_fp4_group_mm, cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data, trtllm_bf16_moe, trtllm_fp8_block_scale_moe, trtllm_fp4_block_scale_moe |
| `self.shared_experts()` | 9 | fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm, int8_scaled_mm, silu_and_mul, gelu_and_mul, gelu_tanh_and_mul |

---

## Non-Model Ops

These ops run outside the main forward pass (pre-processing, post-processing, speculative decoding).

### Input Quantization (Pre-forward)

Called during FP8/INT8 linear layer input quantization.

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `per_token_quant_fp8()` | 1 | sgl_per_token_quant_fp8 |
| `per_token_group_quant_fp8()` | 1 | sgl_per_token_group_quant_fp8 |
| `per_token_group_quant_8bit()` | 1 | sgl_per_token_group_quant_8bit |
| `per_token_group_quant_int8()` | 1 | sgl_per_token_group_quant_int8 |

### NSA (Native Sparse Attention)

Called during NSA indexer and attention.

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `nsa_indexer.forward()` | 1 | hadamard_transform |
| `nsa_backend.forward_decode()` | 2 | fast_topk_v2, fast_topk_transform_fused |
| `nsa_backend.forward_extend()` | 1 | fast_topk_transform_ragged_fused |

### Sampling (Post-forward)

Called during token sampling after model forward.

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `sampler.forward()` | 4 | min_p_sampling_from_probs, top_k_renorm_prob, top_p_renorm_prob, top_k_top_p_sampling_from_probs |

### Speculative Decoding (EAGLE)

Called during EAGLE speculative decoding.

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `eagle_worker.forward()` | 2 | fast_topk, build_tree_kernel_efficient |
| `eagle_utils.verify_tree()` | 1 | verify_tree_greedy |
| `eagle_info.tree_speculative_sampling()` | 1 | tree_speculative_sampling_target_only |

### Utility

| Op | Kernel Count | Primary Kernels |
|----|--------------|-----------------|
| `weak_ref_tensors()` | 1 | weak_ref_tensor |

---

## Cross-Validation Summary

✅ All 60 kernels from `results/deepseek_kernel_traces.md` are mapped to ops above.

| Category | Count | Kernels |
|----------|-------|---------|
| Attention | 7 | cutlass_mla_decode, cutlass_mla_get_workspace_size, trtllm_batch_decode_with_kv_cache_mla, trtllm_ragged_attention_deepseek, merge_state, merge_state_v2, concat_mla_k, concat_mla_absorb_q |
| Activation | 3 | silu_and_mul, gelu_and_mul, gelu_tanh_and_mul |
| Normalization | 2 | rmsnorm, fused_add_rmsnorm |
| GEMM (Linear) | 6 | int8_scaled_mm, fp8_blockwise_scaled_mm, fp8_scaled_mm, cutlass_scaled_fp4_mm, qserve_w4a8_per_chn_gemm, qserve_w4a8_per_group_gemm |
| GEMM (BMM) | 1 | bmm_fp8 |
| RoPE | 3 | apply_rope_with_cos_sin_cache_inplace, mla_rope_quantize_fp8, rotary_embedding |
| MoE Gating | 3 | topk_softmax, topk_sigmoid, moe_fused_gate |
| MoE Compute | 15 | fused_moe_kernel, fp8_blockwise_scaled_grouped_mm, es_fp8_blockwise_scaled_grouped_mm, prepare_moe_input, shuffle_rows, apply_shuffle_mul_sum, moe_align_block_size, moe_sum_reduce, scaled_fp4_experts_quant, cutlass_fp4_group_mm, cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data, trtllm_bf16_moe, trtllm_fp8_block_scale_moe, trtllm_fp4_block_scale_moe |
| DSv3 Low-Latency | 2 | dsv3_fused_a_gemm, dsv3_router_gemm |
| Quantization | 4 | sgl_per_token_quant_fp8, sgl_per_token_group_quant_fp8, sgl_per_token_group_quant_8bit, sgl_per_token_group_quant_int8 |
| NSA | 4 | hadamard_transform, fast_topk_v2, fast_topk_transform_fused, fast_topk_transform_ragged_fused |
| Sampling | 4 | min_p_sampling_from_probs, top_k_renorm_prob, top_p_renorm_prob, top_k_top_p_sampling_from_probs |
| Speculative | 4 | fast_topk, build_tree_kernel_efficient, verify_tree_greedy, tree_speculative_sampling_target_only |
| Utility | 1 | weak_ref_tensor |
| **Total** | **60** | |

---

## Notes

- **Container ops** like `self.model()`, `self.mlp()`, `self.self_attn()` don't have direct kernels - they orchestrate sub-ops
- **Quantization variants** determine which GEMM kernel is used:
  - INT8: `int8_scaled_mm`
  - FP8 blockwise: `fp8_blockwise_scaled_mm`
  - FP8 per-tensor: `fp8_scaled_mm`
  - FP4: `cutlass_scaled_fp4_mm`
  - W4A8 (QoQ): `qserve_w4a8_per_chn_gemm`, `qserve_w4a8_per_group_gemm`
- **Backend selection** (Triton, Cutlass, FlashInfer/TRTLLM) determines which MoE implementation is used
- **Attention path** (MHA, MLA, MLA_FUSED_ROPE) determines which attention kernels and projections are used
- **bmm_fp8** is called directly in model code for MLA weight absorption (4 call sites)
- **dsv3_fused_a_gemm** and **dsv3_router_gemm** are DeepSeek V3 low-latency optimizations called directly in model code
- **Non-model ops** (Quantization, NSA, Sampling, Speculative) are called outside the main forward pass
- Line numbers reference `sglang/python/sglang/srt/models/deepseek_v2.py` unless otherwise noted
