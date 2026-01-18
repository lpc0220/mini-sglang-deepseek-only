# DeepSeek Kernel Traces

Generated: 2026-01-18
DeepSeek Models: [deepseek.py](sglang/python/sglang/srt/models/deepseek.py), [deepseek_v2.py](sglang/python/sglang/srt/models/deepseek_v2.py), [deepseek_nextn.py](sglang/python/sglang/srt/models/deepseek_nextn.py)

---

## Summary

| Kernel | Category | Source | Definition File |
|--------|----------|--------|-----------------|
| cutlass_mla_decode | Attention | sgl-kernel | [attention.py:48](sglang/sgl-kernel/python/sgl_kernel/attention.py#L48) |
| merge_state_v2 | Attention | sgl-kernel | [attention.py:25](sglang/sgl-kernel/python/sgl_kernel/attention.py#L25) |
| concat_mla_absorb_q | Attention | sgl-kernel | [elementwise.py:376](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L376) |
| BatchMLAPagedAttentionWrapper | Attention | flashinfer | [mla.py:142](flashinfer/flashinfer/mla.py#L142) |
| trtllm_batch_decode_with_kv_cache_mla | Attention | flashinfer | [mla.py:522](flashinfer/flashinfer/mla.py#L522) |
| prefill_attention_kernel | Attention | triton | [prefill_attention.py:33](sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py#L33) |
| decode_attention_kernel | Attention | triton | [decode_attention.py:35](sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L35) |
| extend_attention_kernel | Attention | triton | [extend_attention.py:94](sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L94) |
| fused_moe_kernel | MoE | triton | [fused_moe_triton_kernels.py:68](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L68) |
| moe_align_block_size | MoE | sgl-kernel | [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) |
| silu_and_mul | Activation | sgl-kernel | [elementwise.py:172](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L172) |
| gelu_and_mul | Activation | sgl-kernel | [elementwise.py:202](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L202) |
| moe_fused_gate | MoE TopK | sgl-kernel | [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) |
| topk_softmax | MoE TopK | sgl-kernel | [top_k.py](sglang/sgl-kernel/python/sgl_kernel/top_k.py) |
| fp8_scaled_mm | Quantization | sgl-kernel | [gemm.py:29](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L29) |
| fp8_blockwise_scaled_mm | Quantization | sgl-kernel | [gemm.py:19](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L19) |
| int8_scaled_mm | Quantization | sgl-kernel | [gemm.py:8](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L8) |
| sgl_per_token_quant_fp8 | Quantization | sgl-kernel | [gemm.py:149](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L149) |
| cutlass_scaled_fp4_mm | Quantization | sgl-kernel | [gemm.py:157](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L157) |
| scaled_fp4_quant | Quantization | sgl-kernel | [gemm.py:174](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L174) |
| rmsnorm | Normalization | sgl-kernel | [elementwise.py:10](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L10) |
| fused_add_rmsnorm | Normalization | sgl-kernel | [elementwise.py:49](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L49) |
| apply_rope_with_cos_sin_cache_inplace | Rotary | sgl-kernel | [elementwise.py:249](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L249) |
| rotary_embedding | Rotary | sgl-kernel | [elementwise.py:335](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L335) |
| top_k_renorm_prob | Sampling | sgl-kernel | [sampling.py:21](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L21) |
| top_p_renorm_prob | Sampling | sgl-kernel | [sampling.py:69](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L69) |
| top_k_top_p_sampling_from_probs | Sampling | sgl-kernel | [sampling.py:219](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L219) |
| hadamard_transform | NSA | sgl-kernel | [hadamard.py](sglang/sgl-kernel/python/sgl_kernel/hadamard.py) |

---

## Detailed Call Chains

### Attention Kernels

#### MLA Attention (Cutlass Backend)

**Call Chain:**
```
deepseek_v2.py:1553 forward_absorb_core()
  └─> attention_backend_handler.py AttentionBackendRegistry
      └─> cutlass_mla_backend.py:27 CutlassMLABackend
          └─> sgl-kernel: attention.py:48 cutlass_mla_decode()
```

**Links:**
- [deepseek_v2.py:1553](sglang/python/sglang/srt/models/deepseek_v2.py#L1553) `forward_absorb_core()`
- [attention_backend_handler.py](sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py) `AttentionBackendRegistry`
- [cutlass_mla_backend.py:27](sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py#L27) `CutlassMLABackend`
- [attention.py:48](sglang/sgl-kernel/python/sgl_kernel/attention.py#L48) `cutlass_mla_decode()`

**Source:** sgl-kernel

---

#### MLA Attention (FlashInfer Backend)

**Call Chain:**
```
deepseek_v2.py:1553 forward_absorb_core()
  └─> flashinfer_mla_backend.py:49 FlashInferMLABackend
      └─> flashinfer: mla.py:142 BatchMLAPagedAttentionWrapper
          └─> flashinfer: mla.py:522 trtllm_batch_decode_with_kv_cache_mla()
```

**Links:**
- [deepseek_v2.py:1553](sglang/python/sglang/srt/models/deepseek_v2.py#L1553) `forward_absorb_core()`
- [flashinfer_mla_backend.py:49](sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L49) `FlashInferMLABackend`
- [mla.py:142](flashinfer/flashinfer/mla.py#L142) `BatchMLAPagedAttentionWrapper`
- [mla.py:522](flashinfer/flashinfer/mla.py#L522) `trtllm_batch_decode_with_kv_cache_mla()`

**Source:** flashinfer

---

#### MLA Attention (TRT-LLM Backend)

**Call Chain:**
```
deepseek_v2.py:1553 forward_absorb_core()
  └─> trtllm_mla_backend.py:256 TrtllmMLABackend
      └─> sgl-kernel: elementwise.py:376 concat_mla_absorb_q()
      └─> flashinfer: mla.py:522 trtllm_batch_decode_with_kv_cache_mla()
```

**Links:**
- [deepseek_v2.py:1553](sglang/python/sglang/srt/models/deepseek_v2.py#L1553) `forward_absorb_core()`
- [trtllm_mla_backend.py:256](sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py#L256) `TrtllmMLABackend`
- [elementwise.py:376](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L376) `concat_mla_absorb_q()`
- [mla.py:522](flashinfer/flashinfer/mla.py#L522) `trtllm_batch_decode_with_kv_cache_mla()`

**Source:** sgl-kernel + flashinfer

---

#### Triton Prefill Attention

**Call Chain:**
```
deepseek_v2.py forward()
  └─> triton_backend.py TritonAttnBackend
      └─> triton: prefill_attention.py:33 @triton.jit prefill_attention_kernel()
```

**Links:**
- [deepseek_v2.py](sglang/python/sglang/srt/models/deepseek_v2.py) `forward()`
- [triton_backend.py](sglang/python/sglang/srt/layers/attention/triton_backend.py) `TritonAttnBackend`
- [prefill_attention.py:33](sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py#L33) `@triton.jit`

**Source:** triton

---

#### Triton Decode Attention

**Call Chain:**
```
deepseek_v2.py forward()
  └─> triton_backend.py TritonAttnBackend
      └─> triton: decode_attention.py:35 @triton.jit decode_attention_fwd_kernel()
      └─> triton: decode_attention.py:244 @triton.jit decode_attention_fwd_kernel_mqa()
```

**Links:**
- [deepseek_v2.py](sglang/python/sglang/srt/models/deepseek_v2.py) `forward()`
- [triton_backend.py](sglang/python/sglang/srt/layers/attention/triton_backend.py) `TritonAttnBackend`
- [decode_attention.py:35](sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L35) `decode_attention_fwd_kernel()`
- [decode_attention.py:244](sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py#L244) `decode_attention_fwd_kernel_mqa()`

**Source:** triton

---

#### Triton Extend Attention

**Call Chain:**
```
deepseek_v2.py forward()
  └─> triton_backend.py TritonAttnBackend
      └─> triton: extend_attention.py:94 @triton.jit _fwd_kernel()
      └─> triton: extend_attention.py:206 @triton.jit _fwd_kernel_q_inner()
```

**Links:**
- [deepseek_v2.py](sglang/python/sglang/srt/models/deepseek_v2.py) `forward()`
- [triton_backend.py](sglang/python/sglang/srt/layers/attention/triton_backend.py) `TritonAttnBackend`
- [extend_attention.py:94](sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L94) `_fwd_kernel()`
- [extend_attention.py:206](sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L206) `_fwd_kernel_q_inner()`

**Source:** triton

---

#### NSA (Native Sparse Attention)

**Call Chain:**
```
deepseek_v2.py:57 dequantize_k_cache_paged
  └─> nsa_indexer.py:99 Indexer
      └─> sgl-kernel: hadamard.py hadamard_transform()
  └─> nsa_backend.py:182 NSABackend
      └─> triton: triton_kernel.py:9 @triton.jit nsa_triton_kernel()
```

**Links:**
- [deepseek_v2.py:57](sglang/python/sglang/srt/models/deepseek_v2.py#L57) `dequantize_k_cache_paged`
- [nsa_indexer.py:99](sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py#L99) `Indexer`
- [hadamard.py](sglang/sgl-kernel/python/sgl_kernel/hadamard.py) `hadamard_transform()`
- [nsa_backend.py:182](sglang/python/sglang/srt/layers/attention/nsa_backend.py#L182) `NSABackend`
- [triton_kernel.py:9](sglang/python/sglang/srt/layers/attention/nsa/triton_kernel.py#L9) `nsa_triton_kernel()`

**Source:** sgl-kernel + triton

---

#### Attention State Merge

**Call Chain:**
```
merge_state.py:4
  └─> sgl-kernel: attention.py:25 merge_state_v2()
```

**Links:**
- [merge_state.py:4](sglang/python/sglang/srt/layers/attention/merge_state.py#L4)
- [attention.py:25](sglang/sgl-kernel/python/sgl_kernel/attention.py#L25) `merge_state_v2()`

**Source:** sgl-kernel

---

### MoE Kernels

#### Fused MoE (Triton)

**Call Chain:**
```
deepseek_v2.py:96 FusedMoE
  └─> layer.py FusedMoE.forward()
      └─> fused_moe.py invoke_fused_moe_kernel()
          └─> triton: fused_moe_triton_kernels.py:68 @triton.jit moe_gemm_reduce_scatter_triton()
          └─> triton: fused_moe_triton_kernels.py:320 @triton.jit fused_moe_kernel()
```

**Links:**
- [deepseek_v2.py:96](sglang/python/sglang/srt/models/deepseek_v2.py#L96) `FusedMoE`
- [layer.py](sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py) `FusedMoE.forward()`
- [fused_moe.py](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py) `invoke_fused_moe_kernel()`
- [fused_moe_triton_kernels.py:68](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L68) `moe_gemm_reduce_scatter_triton()`
- [fused_moe_triton_kernels.py:320](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L320) `fused_moe_kernel()`

**Source:** triton

---

#### MoE Align Block Size

**Call Chain:**
```
moe_align_block_size.py:13
  └─> sgl-kernel: moe.py moe_align_block_size()
```

**Links:**
- [moe_align_block_size.py:13](sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py#L13)
- [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) `moe_align_block_size()`

**Source:** sgl-kernel

---

#### MoE TopK / Gating

**Call Chain:**
```
deepseek_v2.py DeepseekV2MoE
  └─> topk.py:65 TopK
      └─> sgl-kernel: moe.py moe_fused_gate()
      └─> sgl-kernel: top_k.py topk_softmax()
      └─> sgl-kernel: top_k.py topk_sigmoid()
```

**Links:**
- [deepseek_v2.py](sglang/python/sglang/srt/models/deepseek_v2.py) `DeepseekV2MoE`
- [topk.py:65](sglang/python/sglang/srt/layers/moe/topk.py#L65) `TopK`
- [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) `moe_fused_gate()`
- [top_k.py](sglang/sgl-kernel/python/sgl_kernel/top_k.py) `topk_softmax()`, `topk_sigmoid()`

**Source:** sgl-kernel

---

#### MoE Sum Reduce

**Call Chain:**
```
fused_moe.py:33
  └─> sgl-kernel: moe.py moe_sum_reduce()
```

**Links:**
- [fused_moe.py:33](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L33)
- [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) `moe_sum_reduce()`

**Source:** sgl-kernel

---

#### Cutlass MoE

**Call Chain:**
```
cutlass_moe.py:12
  └─> sgl-kernel: cutlass_moe.py cutlass_moe_gemm()
```

**Links:**
- [cutlass_moe.py:12](sglang/python/sglang/srt/layers/moe/cutlass_moe.py#L12)
- [cutlass_moe.py](sglang/sgl-kernel/python/sgl_kernel/cutlass_moe.py) `cutlass_moe_gemm()`

**Source:** sgl-kernel

---

#### FlashInfer TRT-LLM MoE

**Call Chain:**
```
flashinfer_trtllm.py:127
  └─> flashinfer: fused_moe/ trtllm_fp8_per_tensor_scale_moe()
  └─> flashinfer: fused_moe/ trtllm_bf16_moe()
  └─> flashinfer: fused_moe/ trtllm_fp4_block_scale_moe()
```

**Links:**
- [flashinfer_trtllm.py:127](sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py#L127)
- [fused_moe/](flashinfer/flashinfer/fused_moe/)

**Source:** flashinfer

---

### Activation Kernels

#### SiLU and Mul

**Call Chain:**
```
deepseek_v2.py:56 SiluAndMul
  └─> activation.py:39
      └─> sgl-kernel: elementwise.py:172 silu_and_mul()
```

**Links:**
- [deepseek_v2.py:56](sglang/python/sglang/srt/models/deepseek_v2.py#L56) `SiluAndMul`
- [activation.py:39](sglang/python/sglang/srt/layers/activation.py#L39)
- [elementwise.py:172](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L172) `silu_and_mul()`

**Source:** sgl-kernel

---

#### GELU and Mul

**Call Chain:**
```
activation.py:39
  └─> sgl-kernel: elementwise.py:202 gelu_and_mul()
  └─> sgl-kernel: elementwise.py:187 gelu_tanh_and_mul()
```

**Links:**
- [activation.py:39](sglang/python/sglang/srt/layers/activation.py#L39)
- [elementwise.py:202](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L202) `gelu_and_mul()`
- [elementwise.py:187](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L187) `gelu_tanh_and_mul()`

**Source:** sgl-kernel

---

### Normalization Kernels

#### RMSNorm

**Call Chain:**
```
deepseek_v2.py:82 RMSNorm
  └─> layernorm.py:47
      └─> sgl-kernel: elementwise.py:10 rmsnorm()
      └─> sgl-kernel: elementwise.py:49 fused_add_rmsnorm()
      └─> flashinfer: norm.py layernorm()
```

**Links:**
- [deepseek_v2.py:82](sglang/python/sglang/srt/models/deepseek_v2.py#L82) `RMSNorm`
- [layernorm.py:47](sglang/python/sglang/srt/layers/layernorm.py#L47)
- [elementwise.py:10](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L10) `rmsnorm()`
- [elementwise.py:49](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L49) `fused_add_rmsnorm()`
- [norm.py](flashinfer/flashinfer/norm.py) `layernorm()`

**Source:** sgl-kernel + flashinfer

---

### Rotary Embedding Kernels

#### Rotary Position Embedding

**Call Chain:**
```
deepseek_v2.py:126 get_rope_wrapper
  └─> rotary_embedding.py:19
      └─> sgl-kernel: elementwise.py:249 apply_rope_with_cos_sin_cache_inplace()
      └─> sgl-kernel: elementwise.py:335 rotary_embedding()
```

**Links:**
- [deepseek_v2.py:126](sglang/python/sglang/srt/models/deepseek_v2.py#L126) `get_rope_wrapper`
- [rotary_embedding.py:19](sglang/python/sglang/srt/layers/rotary_embedding.py#L19)
- [elementwise.py:249](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L249) `apply_rope_with_cos_sin_cache_inplace()`
- [elementwise.py:335](sglang/sgl-kernel/python/sgl_kernel/elementwise.py#L335) `rotary_embedding()`

**Source:** sgl-kernel

---

### Quantization Kernels

#### FP8 GEMM

**Call Chain:**
```
deepseek_v2.py:109 Fp8Config
  └─> fp8.py Fp8LinearMethod.apply()
      └─> fp8_utils.py:55
          └─> sgl-kernel: gemm.py:29 fp8_scaled_mm()
          └─> sgl-kernel: gemm.py:19 fp8_blockwise_scaled_mm()
```

**Links:**
- [deepseek_v2.py:109](sglang/python/sglang/srt/models/deepseek_v2.py#L109) `Fp8Config`
- [fp8.py](sglang/python/sglang/srt/layers/quantization/fp8.py) `Fp8LinearMethod.apply()`
- [fp8_utils.py:55](sglang/python/sglang/srt/layers/quantization/fp8_utils.py#L55)
- [gemm.py:29](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L29) `fp8_scaled_mm()`
- [gemm.py:19](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L19) `fp8_blockwise_scaled_mm()`

**Source:** sgl-kernel

---

#### FP8 Per-Token Quantization

**Call Chain:**
```
fp8_kernel.py:39
  └─> sgl-kernel: gemm.py:149 sgl_per_token_quant_fp8()
  └─> sgl-kernel: gemm.py:94 sgl_per_token_group_quant_8bit()
```

**Links:**
- [fp8_kernel.py:39](sglang/python/sglang/srt/layers/quantization/fp8_kernel.py#L39)
- [gemm.py:149](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L149) `sgl_per_token_quant_fp8()`
- [gemm.py:94](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L94) `sgl_per_token_group_quant_8bit()`

**Source:** sgl-kernel

---

#### INT8 GEMM

**Call Chain:**
```
w8a8_int8.py:35
  └─> sgl-kernel: gemm.py:8 int8_scaled_mm()
```

**Links:**
- [w8a8_int8.py:35](sglang/python/sglang/srt/layers/quantization/w8a8_int8.py#L35)
- [gemm.py:8](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L8) `int8_scaled_mm()`

**Source:** sgl-kernel

---

#### FP4 Quantization (ModelOpt/MXFP4)

**Call Chain:**
```
modelopt_quant.py:71
  └─> sgl-kernel: gemm.py:174 scaled_fp4_quant()
  └─> sgl-kernel: gemm.py:157 cutlass_scaled_fp4_mm()
  └─> flashinfer: fp4_quantization.py fp4_quantize()
  └─> flashinfer: quantization.py mm_fp4()
```

**Links:**
- [modelopt_quant.py:71](sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#L71)
- [gemm.py:174](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L174) `scaled_fp4_quant()`
- [gemm.py:157](sglang/sgl-kernel/python/sgl_kernel/gemm.py#L157) `cutlass_scaled_fp4_mm()`
- [fp4_quantization.py](flashinfer/flashinfer/fp4_quantization.py) `fp4_quantize()`
- [quantization.py](flashinfer/flashinfer/quantization.py) `mm_fp4()`

**Source:** sgl-kernel + flashinfer

---

#### MXFP4 Quantization

**Call Chain:**
```
mxfp4.py:60
  └─> flashinfer: quantization.py mxfp4_quantize()
  └─> flashinfer: quantization.py nvfp4_block_scale_interleave()
```

**Links:**
- [mxfp4.py:60](sglang/python/sglang/srt/layers/quantization/mxfp4.py#L60)
- [quantization.py](flashinfer/flashinfer/quantization.py) `mxfp4_quantize()`, `nvfp4_block_scale_interleave()`

**Source:** flashinfer

---

### Sampling Kernels

#### Top-K/Top-P Sampling

**Call Chain:**
```
sampler.py:20
  └─> sgl-kernel: sampling.py:21 top_k_renorm_prob()
  └─> sgl-kernel: sampling.py:69 top_p_renorm_prob()
  └─> sgl-kernel: sampling.py:219 top_k_top_p_sampling_from_probs()
```

**Links:**
- [sampler.py:20](sglang/python/sglang/srt/layers/sampler.py#L20)
- [sampling.py:21](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L21) `top_k_renorm_prob()`
- [sampling.py:69](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L69) `top_p_renorm_prob()`
- [sampling.py:219](sglang/sgl-kernel/python/sgl_kernel/sampling.py#L219) `top_k_top_p_sampling_from_probs()`

**Source:** sgl-kernel

---

### Memory/KV Cache Kernels

#### KV Cache I/O

**Call Chain:**
```
memory_pool_host.py:25
  └─> sgl-kernel: kvcacheio.py
      - load_fp8_kv_cache()
      - store_fp8_kv_cache()
      - load_int8_kv_cache()
      - store_int8_kv_cache()
```

**Links:**
- [memory_pool_host.py:25](sglang/python/sglang/srt/mem_cache/memory_pool_host.py#L25)
- [kvcacheio.py](sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py)

**Source:** sgl-kernel

---

### Communication Kernels

#### Custom All-Reduce

**Call Chain:**
```
custom_all_reduce_ops.py:18
  └─> sgl-kernel: allreduce.py
```

**Links:**
- [custom_all_reduce_ops.py:18](sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py#L18)
- [allreduce.py](sglang/sgl-kernel/python/sgl_kernel/allreduce.py)

**Source:** sgl-kernel

---

#### FlashInfer Communication Fusion

**Call Chain:**
```
flashinfer_comm_fusion.py:18
  └─> flashinfer: comm.py
```

**Links:**
- [flashinfer_comm_fusion.py:18](sglang/python/sglang/srt/layers/flashinfer_comm_fusion.py#L18)
- [comm.py](flashinfer/flashinfer/comm.py)

**Source:** flashinfer

---

## Kernel Source Breakdown

### sgl-kernel ([sglang/sgl-kernel/](sglang/sgl-kernel/))

| Module | Kernels |
|--------|---------|
| [attention.py](sglang/sgl-kernel/python/sgl_kernel/attention.py) | cutlass_mla_decode, cutlass_mla_get_workspace_size, merge_state, merge_state_v2 |
| [elementwise.py](sglang/sgl-kernel/python/sgl_kernel/elementwise.py) | rmsnorm, fused_add_rmsnorm, silu_and_mul, gelu_and_mul, gelu_tanh_and_mul, apply_rope_with_cos_sin_cache_inplace, rotary_embedding, concat_mla_absorb_q, concat_mla_k |
| [gemm.py](sglang/sgl-kernel/python/sgl_kernel/gemm.py) | int8_scaled_mm, fp8_scaled_mm, fp8_blockwise_scaled_mm, sgl_per_token_quant_fp8, sgl_per_token_group_quant_8bit, cutlass_scaled_fp4_mm, scaled_fp4_quant, dsv3_fused_a_gemm, dsv3_router_gemm |
| [moe.py](sglang/sgl-kernel/python/sgl_kernel/moe.py) | moe_align_block_size, moe_sum_reduce, moe_fused_gate |
| [top_k.py](sglang/sgl-kernel/python/sgl_kernel/top_k.py) | topk_softmax, topk_sigmoid |
| [sampling.py](sglang/sgl-kernel/python/sgl_kernel/sampling.py) | top_k_renorm_prob, top_p_renorm_prob, top_k_top_p_sampling_from_probs, min_p_sampling_from_probs |
| [hadamard.py](sglang/sgl-kernel/python/sgl_kernel/hadamard.py) | hadamard_transform |
| [kvcacheio.py](sglang/sgl-kernel/python/sgl_kernel/kvcacheio.py) | load_fp8_kv_cache, store_fp8_kv_cache, load_int8_kv_cache, store_int8_kv_cache |
| [allreduce.py](sglang/sgl-kernel/python/sgl_kernel/allreduce.py) | custom_ar operations |
| [cutlass_moe.py](sglang/sgl-kernel/python/sgl_kernel/cutlass_moe.py) | cutlass_moe_gemm, cutlass_w4a8_moe_mm |

---

### FlashInfer ([flashinfer/](flashinfer/))

| Module | Kernels |
|--------|---------|
| [mla.py](flashinfer/flashinfer/mla.py) | BatchMLAPagedAttentionWrapper, trtllm_batch_decode_with_kv_cache_mla, xqa_batch_decode_with_kv_cache_mla |
| [attention.py](flashinfer/flashinfer/attention.py) | BatchAttention, BatchPrefillWithPagedKVCacheWrapper |
| [decode.py](flashinfer/flashinfer/decode.py) | batch_decode kernels |
| [prefill.py](flashinfer/flashinfer/prefill.py) | batch_prefill kernels |
| [norm.py](flashinfer/flashinfer/norm.py) | layernorm, rmsnorm |
| [fused_moe/](flashinfer/flashinfer/fused_moe/) | trtllm_fp8_per_tensor_scale_moe, trtllm_bf16_moe, trtllm_fp4_block_scale_moe, cutlass_fused_moe |
| [fp4_quantization.py](flashinfer/flashinfer/fp4_quantization.py) | fp4_quantize |
| [fp8_quantization.py](flashinfer/flashinfer/fp8_quantization.py) | fp8_quantize |
| [quantization.py](flashinfer/flashinfer/quantization.py) | mxfp4_quantize, nvfp4_block_scale_interleave |
| [rope.py](flashinfer/flashinfer/rope.py) | rope kernels |
| [comm.py](flashinfer/flashinfer/comm.py) | communication fusion |

---

### Triton JIT ([sglang/python/](sglang/python/))

| File | Kernels |
|------|---------|
| [prefill_attention.py](sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py) | prefill_attention_kernel |
| [decode_attention.py](sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py) | decode_attention_fwd_kernel, decode_attention_fwd_kernel_mqa |
| [extend_attention.py](sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py) | _fwd_kernel, _fwd_kernel_q_inner |
| [triton_kernel.py](sglang/python/sglang/srt/layers/attention/nsa/triton_kernel.py) | nsa_triton_kernel |
| [index_buf_accessor.py](sglang/python/sglang/srt/layers/attention/nsa/index_buf_accessor.py) | index buffer kernels |
| [quant_k_cache.py](sglang/python/sglang/srt/layers/attention/nsa/quant_k_cache.py) | k cache quantization kernels |
| [dequant_k_cache.py](sglang/python/sglang/srt/layers/attention/nsa/dequant_k_cache.py) | k cache dequantization kernels |
| [fused_moe_triton_kernels.py](sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py) | fused_moe_kernel, moe_gemm_reduce_scatter_triton |
| [logits_processor.py](sglang/python/sglang/srt/layers/logits_processor.py) | logits processing kernels |
| [memory_pool.py](sglang/python/sglang/srt/mem_cache/memory_pool.py) | memory management kernels |
| [allocator.py](sglang/python/sglang/srt/mem_cache/allocator.py) | allocator kernels |
| [eagle_info_v2.py](sglang/python/sglang/srt/speculative/eagle_info_v2.py) | speculative decoding kernels |
| [spec_utils.py](sglang/python/sglang/srt/speculative/spec_utils.py) | speculation utility kernels |
| [batch_invariant_ops.py](sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py) | batch invariant matmul kernels |

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
   - [sglang/sgl-kernel/csrc/](sglang/sgl-kernel/csrc/) (sgl-kernel C++/CUDA)
   - [flashinfer/csrc/](flashinfer/csrc/) (flashinfer C++/CUDA)
   - [flashinfer/include/](flashinfer/include/) (flashinfer headers)
