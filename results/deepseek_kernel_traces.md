# DeepSeek Kernel Traces

Generated: 2026-01-18
DeepSeek Models: deepseek.py, deepseek_v2.py, deepseek_nextn.py

---

## Summary

| Kernel | Category | Source | Definition File |
|--------|----------|--------|-----------------|
| cutlass_mla_decode | Attention | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/attention.py:48 |
| merge_state_v2 | Attention | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/attention.py:25 |
| concat_mla_absorb_q | Attention | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:376 |
| BatchMLAPagedAttentionWrapper | Attention | flashinfer | flashinfer/flashinfer/mla.py:142 |
| trtllm_batch_decode_with_kv_cache_mla | Attention | flashinfer | flashinfer/flashinfer/mla.py:522 |
| prefill_attention_kernel | Attention | triton | sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:33 |
| decode_attention_kernel | Attention | triton | sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:35 |
| extend_attention_kernel | Attention | triton | sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:94 |
| fused_moe_kernel | MoE | triton | sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:68 |
| moe_align_block_size | MoE | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/moe.py |
| silu_and_mul | Activation | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:172 |
| gelu_and_mul | Activation | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:202 |
| moe_fused_gate | MoE TopK | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/moe.py |
| topk_softmax | MoE TopK | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/top_k.py |
| fp8_scaled_mm | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:29 |
| fp8_blockwise_scaled_mm | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:19 |
| int8_scaled_mm | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:8 |
| sgl_per_token_quant_fp8 | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:149 |
| cutlass_scaled_fp4_mm | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:157 |
| scaled_fp4_quant | Quantization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/gemm.py:174 |
| rmsnorm | Normalization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:10 |
| fused_add_rmsnorm | Normalization | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:49 |
| apply_rope_with_cos_sin_cache_inplace | Rotary | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:249 |
| rotary_embedding | Rotary | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/elementwise.py:335 |
| top_k_renorm_prob | Sampling | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/sampling.py:21 |
| top_p_renorm_prob | Sampling | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/sampling.py:69 |
| top_k_top_p_sampling_from_probs | Sampling | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/sampling.py:219 |
| hadamard_transform | NSA | sgl-kernel | sglang/sgl-kernel/python/sgl_kernel/hadamard.py |

---

## Detailed Call Chains

### Attention Kernels

#### MLA Attention (Cutlass Backend)

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:1553 forward_absorb_core()
  └─> sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py AttentionBackendRegistry
      └─> sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py:27 CutlassMLABackend
          └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/attention.py:48 cutlass_mla_decode()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/attention.py:48

---

#### MLA Attention (FlashInfer Backend)

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:1553 forward_absorb_core()
  └─> sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:49 FlashInferMLABackend
      └─> flashinfer: flashinfer/flashinfer/mla.py:142 BatchMLAPagedAttentionWrapper
          └─> flashinfer: flashinfer/flashinfer/mla.py:522 trtllm_batch_decode_with_kv_cache_mla()
```

**Source:** flashinfer
**Definition File:** flashinfer/flashinfer/mla.py:142

---

#### MLA Attention (TRT-LLM Backend)

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:1553 forward_absorb_core()
  └─> sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py:256 TrtllmMLABackend
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:376 concat_mla_absorb_q()
      └─> flashinfer: flashinfer/flashinfer/mla.py:522 trtllm_batch_decode_with_kv_cache_mla()
```

**Source:** sgl-kernel + flashinfer
**Definition Files:**
- sglang/sgl-kernel/python/sgl_kernel/elementwise.py:376
- flashinfer/flashinfer/mla.py:522

---

#### Triton Prefill Attention

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py forward()
  └─> sglang/python/sglang/srt/layers/attention/triton_backend.py TritonAttnBackend
      └─> triton: sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:33 @triton.jit prefill_attention_kernel()
```

**Source:** triton
**Definition File:** sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py:33

---

#### Triton Decode Attention

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py forward()
  └─> sglang/python/sglang/srt/layers/attention/triton_backend.py TritonAttnBackend
      └─> triton: sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:35 @triton.jit decode_attention_fwd_kernel()
      └─> triton: sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:244 @triton.jit decode_attention_fwd_kernel_mqa()
```

**Source:** triton
**Definition File:** sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py:35

---

#### Triton Extend Attention

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py forward()
  └─> sglang/python/sglang/srt/layers/attention/triton_backend.py TritonAttnBackend
      └─> triton: sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:94 @triton.jit _fwd_kernel()
      └─> triton: sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:206 @triton.jit _fwd_kernel_q_inner()
```

**Source:** triton
**Definition File:** sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py:94

---

#### NSA (Native Sparse Attention)

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:57 dequantize_k_cache_paged
  └─> sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:99 Indexer
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/hadamard.py hadamard_transform()
  └─> sglang/python/sglang/srt/layers/attention/nsa_backend.py:182 NSABackend
      └─> triton: sglang/python/sglang/srt/layers/attention/nsa/triton_kernel.py:9 @triton.jit nsa_triton_kernel()
```

**Source:** sgl-kernel + triton
**Definition Files:**
- sglang/sgl-kernel/python/sgl_kernel/hadamard.py
- sglang/python/sglang/srt/layers/attention/nsa/triton_kernel.py:9

---

#### Attention State Merge

**Call Chain:**
```
sglang/python/sglang/srt/layers/attention/merge_state.py:4
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/attention.py:25 merge_state_v2()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/attention.py:25

---

### MoE Kernels

#### Fused MoE (Triton)

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:96 FusedMoE
  └─> sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py FusedMoE.forward()
      └─> sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py invoke_fused_moe_kernel()
          └─> triton: sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:68 @triton.jit moe_gemm_reduce_scatter_triton()
          └─> triton: sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:320 @triton.jit fused_moe_kernel()
```

**Source:** triton
**Definition File:** sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py:68

---

#### MoE Align Block Size

**Call Chain:**
```
sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py:13
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/moe.py moe_align_block_size()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/moe.py

---

#### MoE TopK / Gating

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py DeepseekV2MoE
  └─> sglang/python/sglang/srt/layers/moe/topk.py:65 TopK
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/moe.py moe_fused_gate()
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/top_k.py topk_softmax()
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/top_k.py topk_sigmoid()
```

**Source:** sgl-kernel
**Definition Files:**
- sglang/sgl-kernel/python/sgl_kernel/moe.py
- sglang/sgl-kernel/python/sgl_kernel/top_k.py

---

#### MoE Sum Reduce

**Call Chain:**
```
sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:33
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/moe.py moe_sum_reduce()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/moe.py

---

#### Cutlass MoE

**Call Chain:**
```
sglang/python/sglang/srt/layers/moe/cutlass_moe.py:12
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/cutlass_moe.py cutlass_moe_gemm()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/cutlass_moe.py

---

#### FlashInfer TRT-LLM MoE

**Call Chain:**
```
sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:127
  └─> flashinfer: flashinfer/flashinfer/fused_moe trtllm_fp8_per_tensor_scale_moe()
  └─> flashinfer: flashinfer/flashinfer/fused_moe trtllm_bf16_moe()
  └─> flashinfer: flashinfer/flashinfer/fused_moe trtllm_fp4_block_scale_moe()
```

**Source:** flashinfer
**Definition File:** flashinfer/flashinfer/fused_moe/

---

### Activation Kernels

#### SiLU and Mul

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:56 SiluAndMul
  └─> sglang/python/sglang/srt/layers/activation.py:39
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:172 silu_and_mul()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/elementwise.py:172

---

#### GELU and Mul

**Call Chain:**
```
sglang/python/sglang/srt/layers/activation.py:39
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:202 gelu_and_mul()
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:187 gelu_tanh_and_mul()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/elementwise.py:202

---

### Normalization Kernels

#### RMSNorm

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:82 RMSNorm
  └─> sglang/python/sglang/srt/layers/layernorm.py:47
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:10 rmsnorm()
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:49 fused_add_rmsnorm()
      └─> flashinfer: flashinfer/flashinfer/norm.py layernorm()
```

**Source:** sgl-kernel + flashinfer
**Definition Files:**
- sglang/sgl-kernel/python/sgl_kernel/elementwise.py:10
- flashinfer/flashinfer/norm.py

---

### Rotary Embedding Kernels

#### Rotary Position Embedding

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:126 get_rope_wrapper
  └─> sglang/python/sglang/srt/layers/rotary_embedding.py:19
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:249 apply_rope_with_cos_sin_cache_inplace()
      └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/elementwise.py:335 rotary_embedding()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/elementwise.py:249

---

### Quantization Kernels

#### FP8 GEMM

**Call Chain:**
```
sglang/python/sglang/srt/models/deepseek_v2.py:109 Fp8Config
  └─> sglang/python/sglang/srt/layers/quantization/fp8.py Fp8LinearMethod.apply()
      └─> sglang/python/sglang/srt/layers/quantization/fp8_utils.py:55
          └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:29 fp8_scaled_mm()
          └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:19 fp8_blockwise_scaled_mm()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/gemm.py:29

---

#### FP8 Per-Token Quantization

**Call Chain:**
```
sglang/python/sglang/srt/layers/quantization/fp8_kernel.py:39
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:149 sgl_per_token_quant_fp8()
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:94 sgl_per_token_group_quant_8bit()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/gemm.py:149

---

#### INT8 GEMM

**Call Chain:**
```
sglang/python/sglang/srt/layers/quantization/w8a8_int8.py:35
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:8 int8_scaled_mm()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/gemm.py:8

---

#### FP4 Quantization (ModelOpt/MXFP4)

**Call Chain:**
```
sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:71
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:174 scaled_fp4_quant()
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/gemm.py:157 cutlass_scaled_fp4_mm()
  └─> flashinfer: flashinfer/flashinfer/fp4_quantization.py fp4_quantize()
  └─> flashinfer: flashinfer/flashinfer/quantization.py mm_fp4()
```

**Source:** sgl-kernel + flashinfer
**Definition Files:**
- sglang/sgl-kernel/python/sgl_kernel/gemm.py:174
- flashinfer/flashinfer/fp4_quantization.py

---

#### MXFP4 Quantization

**Call Chain:**
```
sglang/python/sglang/srt/layers/quantization/mxfp4.py:60
  └─> flashinfer: flashinfer/flashinfer/quantization.py mxfp4_quantize()
  └─> flashinfer: flashinfer/flashinfer/quantization.py nvfp4_block_scale_interleave()
```

**Source:** flashinfer
**Definition File:** flashinfer/flashinfer/quantization.py

---

### Sampling Kernels

#### Top-K/Top-P Sampling

**Call Chain:**
```
sglang/python/sglang/srt/layers/sampler.py:20
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/sampling.py:21 top_k_renorm_prob()
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/sampling.py:69 top_p_renorm_prob()
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/sampling.py:219 top_k_top_p_sampling_from_probs()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/sampling.py

---

### Memory/KV Cache Kernels

#### KV Cache I/O

**Call Chain:**
```
sglang/python/sglang/srt/mem_cache/memory_pool_host.py:25
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py
      - load_fp8_kv_cache()
      - store_fp8_kv_cache()
      - load_int8_kv_cache()
      - store_int8_kv_cache()
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py

---

### Communication Kernels

#### Custom All-Reduce

**Call Chain:**
```
sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py:18
  └─> sgl-kernel: sglang/sgl-kernel/python/sgl_kernel/allreduce.py
```

**Source:** sgl-kernel
**Definition File:** sglang/sgl-kernel/python/sgl_kernel/allreduce.py

---

#### FlashInfer Communication Fusion

**Call Chain:**
```
sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py:18
  └─> flashinfer: flashinfer/flashinfer/comm.py
```

**Source:** flashinfer
**Definition File:** flashinfer/flashinfer/comm.py

---

## Kernel Source Breakdown

### sgl-kernel (sglang/sgl-kernel/)

| Module | Kernels |
|--------|---------|
| attention.py | cutlass_mla_decode, cutlass_mla_get_workspace_size, merge_state, merge_state_v2 |
| elementwise.py | rmsnorm, fused_add_rmsnorm, silu_and_mul, gelu_and_mul, gelu_tanh_and_mul, apply_rope_with_cos_sin_cache_inplace, rotary_embedding, concat_mla_absorb_q, concat_mla_k |
| gemm.py | int8_scaled_mm, fp8_scaled_mm, fp8_blockwise_scaled_mm, sgl_per_token_quant_fp8, sgl_per_token_group_quant_8bit, cutlass_scaled_fp4_mm, scaled_fp4_quant, dsv3_fused_a_gemm, dsv3_router_gemm |
| moe.py | moe_align_block_size, moe_sum_reduce, moe_fused_gate |
| top_k.py | topk_softmax, topk_sigmoid |
| sampling.py | top_k_renorm_prob, top_p_renorm_prob, top_k_top_p_sampling_from_probs, min_p_sampling_from_probs |
| hadamard.py | hadamard_transform |
| kvcacheio.py | load_fp8_kv_cache, store_fp8_kv_cache, load_int8_kv_cache, store_int8_kv_cache |
| allreduce.py | custom_ar operations |
| cutlass_moe.py | cutlass_moe_gemm, cutlass_w4a8_moe_mm |

---

### FlashInfer (flashinfer/)

| Module | Kernels |
|--------|---------|
| mla.py | BatchMLAPagedAttentionWrapper, trtllm_batch_decode_with_kv_cache_mla, xqa_batch_decode_with_kv_cache_mla |
| attention.py | BatchAttention, BatchPrefillWithPagedKVCacheWrapper |
| decode.py | batch_decode kernels |
| prefill.py | batch_prefill kernels |
| norm.py | layernorm, rmsnorm |
| fused_moe/ | trtllm_fp8_per_tensor_scale_moe, trtllm_bf16_moe, trtllm_fp4_block_scale_moe, cutlass_fused_moe |
| fp4_quantization.py | fp4_quantize |
| fp8_quantization.py | fp8_quantize |
| quantization.py | mxfp4_quantize, nvfp4_block_scale_interleave |
| rope.py | rope kernels |
| comm.py | communication fusion |

---

### Triton JIT (sglang/python/)

| File | Kernels |
|------|---------|
| layers/attention/triton_ops/prefill_attention.py | prefill_attention_kernel |
| layers/attention/triton_ops/decode_attention.py | decode_attention_fwd_kernel, decode_attention_fwd_kernel_mqa |
| layers/attention/triton_ops/extend_attention.py | _fwd_kernel, _fwd_kernel_q_inner |
| layers/attention/nsa/triton_kernel.py | nsa_triton_kernel |
| layers/attention/nsa/index_buf_accessor.py | index buffer kernels |
| layers/attention/nsa/quant_k_cache.py | k cache quantization kernels |
| layers/attention/nsa/dequant_k_cache.py | k cache dequantization kernels |
| layers/moe/fused_moe_triton/fused_moe_triton_kernels.py | fused_moe_kernel, moe_gemm_reduce_scatter_triton |
| layers/logits_processor.py | logits processing kernels |
| mem_cache/memory_pool.py | memory management kernels |
| mem_cache/allocator.py | allocator kernels |
| speculative/eagle_info_v2.py | speculative decoding kernels |
| speculative/spec_utils.py | speculation utility kernels |
| batch_invariant_ops/batch_invariant_ops.py | batch invariant matmul kernels |

---

## Notes

1. **Primary Attention Backend:** DeepSeek V2/V3/R1 uses MLA (Multi-head Latent Attention) with multiple backend options:
   - Cutlass MLA (sgl-kernel) - Default for NVIDIA
   - FlashInfer MLA - Alternative backend
   - TRT-LLM MLA - High-performance option
   - Triton - Fallback/reference implementation

2. **MoE Execution:** DeepSeek MoE layers can use:
   - Triton fused MoE kernels (default)
   - Cutlass MoE (sgl-kernel)
   - FlashInfer TRT-LLM MoE

3. **Quantization:** DeepSeek-R1 NVFP4 uses:
   - FP8 quantization (sgl-kernel + flashinfer)
   - FP4/MXFP4 quantization (flashinfer)
   - INT8 quantization (sgl-kernel)

4. **All CUDA kernel sources are in:**
   - sglang/sgl-kernel/csrc/ (sgl-kernel C++/CUDA)
   - flashinfer/csrc/ (flashinfer C++/CUDA)
   - flashinfer/include/ (flashinfer headers)
