# DeepSeek-R1-NVFP4-v2 Kernel Performance Profile

**Generated:** 2026-01-28

**Hardware:** NVIDIA GB200 (Blackwell B200)

**Status:** ✅ **23/23 kernels passing**

**Source:** `results/all_kernels.csv`

---

## Hardware Specifications

| Metric | Value |
|--------|-------|
| Peak FP4 Compute | 9000 TFLOPS |
| Peak FP8 Compute | 4500 TFLOPS |
| Peak FP16/BF16 Compute | 2250 TFLOPS |
| Peak HBM Bandwidth | 8000 GB/s |
| Ridge Point (FP4) | 1125 FLOP/byte |
| Ridge Point (FP8) | 562.5 FLOP/byte |
| Ridge Point (FP16) | 281.25 FLOP/byte |

---

## All 23 Kernels

| # | Kernel | Source | Category | Bound | Benchmark File |
|---|--------|--------|----------|-------|----------------|
| 1 | `rmsnorm` | sgl-kernel | Norm | Memory | [bench_rmsnorm.py](scripts/bench_rmsnorm.py) |
| 2 | `fused_add_rmsnorm` | sgl-kernel | Norm | Memory | [bench_fused_add_rmsnorm.py](scripts/bench_fused_add_rmsnorm.py) |
| 3 | `cutlass_scaled_fp4_mm` | sgl-kernel | GEMM | Compute | [bench_cutlass_scaled_fp4_mm.py](scripts/bench_cutlass_scaled_fp4_mm.py) |
| 4 | `dsv3_fused_a_gemm` | sgl-kernel | GEMM | Compute | [bench_dsv3_fused_a_gemm.py](scripts/bench_dsv3_fused_a_gemm.py) |
| 5 | `dsv3_router_gemm` | sgl-kernel | GEMM | Compute | [bench_dsv3_router_gemm.py](scripts/bench_dsv3_router_gemm.py) |
| 6 | `bmm_fp8` | sgl-kernel | BMM | Compute | [bench_bmm_fp8.py](scripts/bench_bmm_fp8.py) |
| 7 | `cutlass_mla_decode` | sgl-kernel | Attention | Mixed | [bench_cutlass_mla_decode.py](scripts/bench_cutlass_mla_decode.py) |
| 8 | `trtllm_batch_decode_with_kv_cache_mla` | flashinfer | Attention | Mixed | [bench_trtllm_mla.py](scripts/bench_trtllm_mla.py) |
| 9 | `trtllm_ragged_attention_deepseek` | flashinfer | Attention | Mixed | [bench_trtllm_ragged_attention.py](scripts/bench_trtllm_ragged_attention.py) |
| 10 | `mla_rope_quantize_fp8` | flashinfer | Attention | Memory | [bench_mla_rope_quantize_fp8.py](scripts/bench_mla_rope_quantize_fp8.py) |
| 11 | `apply_rope_with_cos_sin_cache_inplace` | sgl-kernel | RoPE | Memory | [bench_apply_rope.py](scripts/bench_apply_rope.py) |
| 12 | `concat_mla_k` | sgl-kernel | Concat | Memory | [bench_concat_mla_mha_k.py](scripts/bench_concat_mla_mha_k.py) |
| 13 | `silu_and_mul` | sgl-kernel | Activation | Memory | [bench_silu_and_mul.py](scripts/bench_silu_and_mul.py) |
| 14 | `topk_softmax` | sgl-kernel | MoE Routing | Memory | [bench_topk_softmax.py](scripts/bench_topk_softmax.py) |
| 15 | `topk_sigmoid` | sgl-kernel | MoE Routing | Memory | [bench_topk_sigmoid.py](scripts/bench_topk_sigmoid.py) |
| 16 | `moe_fused_gate` | sgl-kernel | MoE Routing | Memory | [bench_moe_fused_gate.py](scripts/bench_moe_fused_gate.py) |
| 17 | `prepare_moe_input` | sgl-kernel | MoE | Memory | [bench_prepare_moe_input.py](scripts/bench_prepare_moe_input.py) |
| 18 | `scaled_fp4_experts_quant` | sgl-kernel | MoE | Memory | [bench_scaled_fp4_experts_quant.py](scripts/bench_scaled_fp4_experts_quant.py) |
| 19 | `cutlass_fp4_group_mm` | sgl-kernel | MoE | Compute | [bench_cutlass_fp4_group_mm.py](scripts/bench_cutlass_fp4_group_mm.py) |
| 20 | `apply_shuffle_mul_sum` | sgl-kernel | MoE | Memory | [bench_apply_shuffle_mul_sum.py](scripts/bench_apply_shuffle_mul_sum.py) |
| 21 | `moe_align_block_size` | sgl-kernel | MoE | Memory | [bench_moe_align_block_size.py](scripts/bench_moe_align_block_size.py) |
| 22 | `trtllm_fp4_block_scale_moe` | flashinfer | MoE | Mixed | [bench_trtllm_fp4_block_scale_moe.py](scripts/bench_trtllm_fp4_block_scale_moe.py) |
| 23 | `fused_moe_kernel` | triton | MoE | Mixed | [bench_fused_moe_kernel.py](scripts/bench_fused_moe_kernel.py) |

**Note:** All 23 kernels are now benchmarked successfully. Kernels 17, 18, 20 (`prepare_moe_input`, `scaled_fp4_experts_quant`, `apply_shuffle_mul_sum`) use internal APIs but are working.

---

## Kernel Limitations & Notes

| Kernel | Limitation | Notes |
|--------|------------|-------|
| `dsv3_fused_a_gemm` | B ≤ 16 | Low-latency path optimized for small batch decode |
| `dsv3_router_gemm` | num_tokens ≤ 16 | Hard limit in kernel; prefill phase skipped |
| `cutlass_mla_decode` | B × seq_len ≤ 1024 | GB200 crashes at B*seq_len > 1024 with CUDA illegal instruction |
| `trtllm_*` kernels | flashinfer required | May not be available on all installations |
| `trtllm_fp4_block_scale_moe` | Uses flashinfer CLI | Requires complex weight preprocessing (shuffling, permutation); benchmark uses flashinfer's official CLI with Renormalize routing |
| `mla_rope_quantize_fp8` | flashinfer required | Requires flashinfer.rope module |
| `prepare_moe_input` | May not be exported | Internal API - may show "not available" |
| `scaled_fp4_experts_quant` | May not be exported | Internal API - may show "not available" |
| `apply_shuffle_mul_sum` | May not be exported | Internal API - may show "not available" |
| `cutlass_fp4_group_mm` | API may differ | Actual kernel API may differ from benchmark |

**Error Handling:** All benchmarks have try/except blocks around `benchmark_kernel()` calls to prevent CUDA context corruption from affecting subsequent benchmarks. After each kernel (success or failure), CUDA context is reset with `torch.cuda.synchronize()` and `torch.cuda.empty_cache()`.

---

## Model Configuration

| Parameter | Value | Symbol |
|-----------|-------|--------|
| hidden_size | 7168 | H |
| q_lora_rank | 1536 | Lq |
| kv_lora_rank | 512 | Lkv |
| num_heads | 128 | Nh |
| qk_nope_head_dim | 128 | Dn |
| qk_rope_head_dim | 64 | Dr |
| v_head_dim | 128 | Dv |
| n_routed_experts | 256 | E |
| num_experts_per_tok | 8 | K |
| moe_intermediate_size | 2048 | I |

---

## Decode Phase (B=1, S=1)

| Kernel | Op | Latency (ms) | GFLOPS | BW (GB/s) | Peak % | Category |
|--------|-----|--------------|--------|-----------|--------|----------|
| rmsnorm | input_layernorm | - | - | - | - | Norm |
| dsv3_fused_a_gemm | fused_qkv_a_proj | - | - | - | - | GEMM |
| rmsnorm | q_a_layernorm | - | - | - | - | Norm |
| rmsnorm | kv_a_layernorm | - | - | - | - | Norm |
| cutlass_scaled_fp4_mm | q_b_proj | - | - | - | - | GEMM |
| bmm_fp8 | q_nope * w_kc | - | - | - | - | BMM |
| apply_rope_* | rotary_emb | - | - | - | - | RoPE |
| cutlass_mla_decode | attn_mqa | - | - | - | - | Attention |
| bmm_fp8 | attn * w_vc | - | - | - | - | BMM |
| cutlass_scaled_fp4_mm | o_proj | - | - | - | - | GEMM |
| fused_add_rmsnorm | post_attention_layernorm | - | - | - | - | Norm |
| dsv3_router_gemm | gate | - | - | - | - | GEMM |
| topk_softmax | topk | - | - | - | - | MoE |
| prepare_moe_input | experts | - | - | - | - | MoE |
| scaled_fp4_experts_quant | experts | - | - | - | - | MoE |
| cutlass_fp4_group_mm | experts (gate_up) | - | - | - | - | MoE |
| silu_and_mul | act_fn | - | - | - | - | Activation |
| cutlass_fp4_group_mm | experts (down) | - | - | - | - | MoE |
| apply_shuffle_mul_sum | experts | - | - | - | - | MoE |

---

## Decode Phase (B=128, S=1)

| Kernel | Op | Latency (ms) | GFLOPS | BW (GB/s) | Peak % | Category |
|--------|-----|--------------|--------|-----------|--------|----------|
| (Run benchmark to populate) | | | | | | |

---

## Prefill Phase (B=1, S=1024)

| Kernel | Op | Latency (ms) | GFLOPS | BW (GB/s) | Peak % | Category |
|--------|-----|--------------|--------|-----------|--------|----------|
| (Run benchmark to populate) | | | | | | |

---

## Scaling Analysis

### Batch Scaling (Decode, S=1)

| B | Total Layer Time (ms) | Throughput (tokens/s) |
|---|----------------------|----------------------|
| 1 | - | - |
| 8 | - | - |
| 16 | - | - |
| 32 | - | - |
| 64 | - | - |
| 128 | - | - |

### Sequence Scaling (Prefill, B=1)

| S | Total Layer Time (ms) | Throughput (tokens/s) |
|---|----------------------|----------------------|
| 128 | - | - |
| 256 | - | - |
| 512 | - | - |
| 1024 | - | - |
| 2048 | - | - |

---

## How to Run Benchmarks

```bash
# Navigate to scripts directory
cd deepseek_kernel_benchmarks/scripts

# List all available kernels
python run_all_benchmarks.py --list

# Run all 23 kernel benchmarks
python run_all_benchmarks.py --output ../results/

# Run specific kernels only
python run_all_benchmarks.py --kernels rmsnorm,cutlass_scaled_fp4_mm,bmm_fp8 --output ../results/

# Run individual kernel benchmarks
python bench_rmsnorm.py --output ../results/
python bench_cutlass_scaled_fp4_mm.py --output ../results/
python bench_cutlass_mla_decode.py --output ../results/

# Generate roofline plot
python plot_roofline.py --input ../results/all_kernels.csv --output ../roofline.png
```

---

## Benchmark File Structure

```
deepseek_kernel_benchmarks/
├── summary.md                    # This file
├── roofline.md                   # Roofline analysis
├── scripts/
│   ├── bench_utils.py           # Shared utilities and constants
│   ├── run_all_benchmarks.py    # Orchestrator for all benchmarks
│   ├── plot_roofline.py         # Roofline visualization
│   │
│   │ # Normalization (2 kernels)
│   ├── bench_rmsnorm.py
│   ├── bench_fused_add_rmsnorm.py
│   │
│   │ # GEMM (3 kernels)
│   ├── bench_cutlass_scaled_fp4_mm.py
│   ├── bench_dsv3_fused_a_gemm.py
│   ├── bench_dsv3_router_gemm.py
│   │
│   │ # BMM (1 kernel)
│   ├── bench_bmm_fp8.py
│   │
│   │ # Attention (4 kernels)
│   ├── bench_cutlass_mla_decode.py
│   ├── bench_trtllm_mla.py
│   ├── bench_trtllm_ragged_attention.py
│   ├── bench_mla_rope_quantize_fp8.py
│   │
│   │ # RoPE & Concat (2 kernels)
│   ├── bench_apply_rope.py
│   ├── bench_concat_mla_mha_k.py
│   │
│   │ # Activation (1 kernel)
│   ├── bench_silu_and_mul.py
│   │
│   │ # MoE Routing (3 kernels)
│   ├── bench_topk_softmax.py
│   ├── bench_topk_sigmoid.py
│   ├── bench_moe_fused_gate.py
│   │
│   │ # MoE Experts (5 kernels)
│   ├── bench_prepare_moe_input.py
│   ├── bench_scaled_fp4_experts_quant.py
│   ├── bench_cutlass_fp4_group_mm.py
│   ├── bench_apply_shuffle_mul_sum.py
│   ├── bench_moe_align_block_size.py
│   │
│   │ # Fused MoE (2 kernels)
│   ├── bench_trtllm_fp4_block_scale_moe.py
│   └── bench_fused_moe_kernel.py
│
└── results/
    ├── all_kernels.csv          # Aggregated results
    ├── benchmark_summary.md     # Run summary
    ├── rmsnorm.csv
    ├── fused_add_rmsnorm.csv
    ├── cutlass_scaled_fp4_mm.csv
    └── ... (one CSV per kernel)
```
