# DeepSeek-R1-NVFP4-v2 Kernel Benchmark Summary

**Generated:** 2026-01-26 09:51:46

**Total Kernels:** 23
**Kernels Run:** 11
**Kernels Failed:** 12

## Kernel Status

| # | Kernel | Category | Bound | Status |
|---|--------|----------|-------|--------|
| 1 | `rmsnorm` | Norm | Memory | OK |
| 2 | `fused_add_rmsnorm` | Norm | Memory | OK |
| 3 | `cutlass_scaled_fp4_mm` | GEMM | Compute | OK |
| 4 | `dsv3_fused_a_gemm` | GEMM | Compute | OK |
| 5 | `dsv3_router_gemm` | GEMM | Compute | OK |
| 6 | `bmm_fp8` | BMM | Compute | OK |
| 7 | `cutlass_mla_decode` | Attention | Mixed | FAILED |
| 8 | `trtllm_batch_decode_with_kv_cache_mla` | Attention | Mixed | OK |
| 9 | `trtllm_ragged_attention_deepseek` | Attention | Mixed | OK |
| 10 | `mla_rope_quantize_fp8` | Attention | Memory | OK |
| 11 | `apply_rope_with_cos_sin_cache_inplace` | RoPE | Memory | FAILED |
| 12 | `concat_mla_mha_k` | Concat | Memory | OK |
| 13 | `silu_and_mul` | Activation | Memory | FAILED |
| 14 | `topk_softmax` | MoE Routing | Memory | FAILED |
| 15 | `topk_sigmoid` | MoE Routing | Memory | FAILED |
| 16 | `moe_fused_gate` | MoE Routing | Memory | FAILED |
| 17 | `prepare_moe_input` | MoE | Memory | FAILED |
| 18 | `scaled_fp4_experts_quant` | MoE | Memory | FAILED |
| 19 | `cutlass_fp4_group_mm` | MoE | Compute | FAILED |
| 20 | `apply_shuffle_mul_sum` | MoE | Memory | FAILED |
| 21 | `moe_align_block_size` | MoE | Memory | FAILED |
| 22 | `trtllm_fp4_block_scale_moe` | MoE | Mixed | OK |
| 23 | `fused_moe_kernel` | MoE | Mixed | FAILED |

## CSV Files Generated

- `all_kernels.csv`
- `bmm_fp8.csv`
- `concat_mla_mha_k.csv`
- `cutlass_scaled_fp4_mm.csv`
- `dsv3_fused_a_gemm.csv`
- `dsv3_router_gemm.csv`
- `fused_add_rmsnorm.csv`
- `mla_rope_quantize_fp8.csv`
- `rmsnorm.csv`
- `trtllm_batch_decode_with_kv_cache_mla.csv`
- `trtllm_ragged_attention_deepseek.csv`
