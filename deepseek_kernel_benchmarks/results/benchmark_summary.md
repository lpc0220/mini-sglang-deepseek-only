# DeepSeek-R1-NVFP4-v2 Kernel Benchmark Summary

**Generated:** 2026-01-27 21:53:35

**Total Kernels:** 23
**Kernels Run:** 1
**Successful:** 0
**Failed:** 1

**Note:** Each kernel runs in a separate subprocess for isolation.
CUDA crashes in one kernel do not affect other kernels.

## Kernel Status

| # | Kernel | Category | Bound | Status | Notes |
|---|--------|----------|-------|--------|-------|
| 1 | `trtllm_fp4_block_scale_moe` | MoE | Mixed | ✗ FAILED | No CSV output (kernel not available or all runs fa... |

## CSV Files Generated

- `all_kernels.csv`
- `apply_rope_with_cos_sin_cache_inplace.csv`
- `apply_shuffle_mul_sum.csv`
- `bmm_fp8.csv`
- `concat_mla_k.csv`
- `cutlass_fp4_group_mm.csv`
- `cutlass_mla_decode.csv`
- `cutlass_scaled_fp4_mm.csv`
- `dsv3_fused_a_gemm.csv`
- `dsv3_router_gemm.csv`
- `fused_add_rmsnorm.csv`
- `fused_moe_kernel.csv`
- `mla_rope_quantize_fp8.csv`
- `moe_align_block_size.csv`
- `moe_fused_gate.csv`
- `prepare_moe_input.csv`
- `rmsnorm.csv`
- `scaled_fp4_experts_quant.csv`
- `silu_and_mul.csv`
- `topk_sigmoid.csv`
- `topk_softmax.csv`
- `trtllm_batch_decode_with_kv_cache_mla.csv`
- `trtllm_ragged_attention_deepseek.csv`
