# DeepSeek-R1-NVFP4-v2 Kernel Benchmark Summary

**Generated:** 2026-01-26 11:17:50

**Total Kernels:** 23
**Kernels Run:** 23
**Successful:** 12
**Failed:** 11

**Note:** Each kernel runs in a separate subprocess for isolation.
CUDA crashes in one kernel do not affect other kernels.

## Kernel Status

| # | Kernel | Category | Bound | Status | Notes |
|---|--------|----------|-------|--------|-------|
| 1 | `rmsnorm` | Norm | Memory | ✓ OK | OK |
| 2 | `fused_add_rmsnorm` | Norm | Memory | ✓ OK | OK |
| 3 | `cutlass_scaled_fp4_mm` | GEMM | Compute | ✓ OK | OK |
| 4 | `dsv3_fused_a_gemm` | GEMM | Compute | ✓ OK | OK |
| 5 | `dsv3_router_gemm` | GEMM | Compute | ✓ OK | OK |
| 6 | `bmm_fp8` | BMM | Compute | ✓ OK | OK |
| 7 | `cutlass_mla_decode` | Attention | Mixed | ✓ OK | OK |
| 8 | `trtllm_batch_decode_with_kv_cache_mla` | Attention | Mixed | ✓ OK | OK |
| 9 | `trtllm_ragged_attention_deepseek` | Attention | Mixed | ✓ OK | OK |
| 10 | `mla_rope_quantize_fp8` | Attention | Memory | ✗ FAILED | Exit code 2 |
| 11 | `apply_rope_with_cos_sin_cache_inplace` | RoPE | Memory | ✗ FAILED | CSV empty (no successful runs) |
| 12 | `concat_mla_k` | Concat | Memory | ✓ OK | OK |
| 13 | `silu_and_mul` | Activation | Memory | ✓ OK | OK |
| 14 | `topk_softmax` | MoE Routing | Memory | ✗ FAILED | Exit code 1 |
| 15 | `topk_sigmoid` | MoE Routing | Memory | ✗ FAILED | Exit code 1 |
| 16 | `moe_fused_gate` | MoE Routing | Memory | ✓ OK | OK |
| 17 | `prepare_moe_input` | MoE | Memory | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 18 | `scaled_fp4_experts_quant` | MoE | Memory | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 19 | `cutlass_fp4_group_mm` | MoE | Compute | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 20 | `apply_shuffle_mul_sum` | MoE | Memory | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 21 | `moe_align_block_size` | MoE | Memory | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 22 | `trtllm_fp4_block_scale_moe` | MoE | Mixed | ✗ FAILED | No CSV output (kernel not available or all runs fa... |
| 23 | `fused_moe_kernel` | MoE | Mixed | ✗ FAILED | No CSV output (kernel not available or all runs fa... |

## CSV Files Generated

- `all_kernels.csv`
- `apply_rope_with_cos_sin_cache_inplace.csv`
- `bmm_fp8.csv`
- `concat_mla_k.csv`
- `cutlass_mla_decode.csv`
- `cutlass_scaled_fp4_mm.csv`
- `dsv3_fused_a_gemm.csv`
- `dsv3_router_gemm.csv`
- `fused_add_rmsnorm.csv`
- `mla_rope_quantize_fp8.csv`
- `moe_fused_gate.csv`
- `rmsnorm.csv`
- `silu_and_mul.csv`
- `topk_sigmoid.csv`
- `topk_softmax.csv`
- `trtllm_batch_decode_with_kv_cache_mla.csv`
- `trtllm_ragged_attention_deepseek.csv`
