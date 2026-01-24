# GB200 Roofline Analysis - DeepSeek-R1-NVFP4-v2

**Generated:** [Run benchmark to populate]

---

## Hardware Limits

| Metric | Value |
|--------|-------|
| Peak Compute (FP4) | 9000 TFLOPS |
| Peak Compute (FP8) | 4500 TFLOPS |
| Peak Compute (FP16) | 2250 TFLOPS |
| Peak Bandwidth | 8000 GB/s |
| Ridge Point (FP4) | 1125 FLOP/byte |
| Ridge Point (FP8) | 562.5 FLOP/byte |
| Ridge Point (FP16) | 281.25 FLOP/byte |

---

## Roofline Model

```
Performance = min(Peak_FLOPS, Arithmetic_Intensity × Peak_Bandwidth)

Where:
  Arithmetic Intensity (AI) = FLOPS / Bytes_transferred

  If AI < Ridge_Point: Memory-bound (limited by bandwidth)
  If AI > Ridge_Point: Compute-bound (limited by FLOPS)
```

---

## Kernel Placement

| Kernel | Op | AI (FLOP/byte) | Bound | Attainable (TFLOPS) | Notes |
|--------|-----|----------------|-------|---------------------|-------|
| `cutlass_scaled_fp4_mm` | fused_qkv_a_proj | ~7000 | Compute | 9000 | FP4 GEMM [B, 7168] → [B, 2112] |
| `cutlass_scaled_fp4_mm` | q_b_proj | ~12000 | Compute | 9000 | FP4 GEMM [B, 1536] → [B, 24576] |
| `cutlass_scaled_fp4_mm` | kv_b_proj | ~15000 | Compute | 9000 | FP4 GEMM [B, 512] → [B, 32768] |
| `cutlass_scaled_fp4_mm` | o_proj | ~9000 | Compute | 9000 | FP4 GEMM [B, 16384] → [B, 7168] |
| `bmm_fp8` | q_nope * w_kc | ~256 | Compute | 4500 | FP8 BMM [128, B, 128] × [128, 128, 512] |
| `bmm_fp8` | attn * w_vc | ~256 | Compute | 4500 | FP8 BMM [128, B, 512] × [128, 512, 128] |
| `cutlass_mla_decode` | attn_mqa | varies | Mixed | varies | Depends on seq_len |
| `dsv3_fused_a_gemm` | fused_qkv_a_proj | ~500 | Mixed | 2250 | BF16 low-latency (B≤16) |
| `rmsnorm` | layernorm | ~1.25 | Memory | 10 GFLOPS | 5 ops/element |
| `silu_and_mul` | act_fn | ~0.67 | Memory | 5.4 GFLOPS | 4 ops/element |
| `topk_softmax` | topk | ~2 | Memory | 16 GFLOPS | Routing overhead |

---

## Arithmetic Intensity Formulas

### GEMM (FP4 weights, FP16 activations)

```
FLOPS = 2 × M × N × K
Bytes = M×K×2 + K×N×0.5 + M×N×2  (FP16 input, FP4 weight, FP16 output)
AI = 2×M×N×K / (M×K×2 + K×N×0.5 + M×N×2)

For large K, N >> M:
  AI ≈ 2×M×N×K / (K×N×0.5) = 4×M

Example: q_b_proj with M=128 (batch), K=1536, N=24576
  FLOPS = 2 × 128 × 24576 × 1536 = 9.7e9
  Bytes = 128×1536×2 + 1536×24576×0.5 + 128×24576×2 = 26.1e6
  AI = 371 FLOP/byte → Compute-bound
```

### BMM (FP8)

```
FLOPS = 2 × batch × M × N × K
Bytes = batch × (M×K + K×N + M×N) × 1  (FP8 = 1 byte)
AI = 2×M×N×K / (M×K + K×N + M×N)

Example: q_nope * w_kc with batch=128, M=B, K=128, N=512
  For B=1: AI = 2×1×512×128 / (1×128 + 128×512 + 1×512) = 1.96 FLOP/byte → Memory-bound
  For B=64: AI = 2×64×512×128 / (64×128 + 128×512 + 64×512) = 77 FLOP/byte → Compute-bound
```

### RMSNorm

```
FLOPS ≈ 5 × N  (mul, sub, rsqrt, mul, add per element)
Bytes = 2 × N × 2  (read + write, FP16)
AI = 5N / (4N) = 1.25 FLOP/byte → Memory-bound
```

### SiLU Activation

```
FLOPS ≈ 4 × N  (sigmoid, mul, mul per element)
Bytes = 3 × N × 2  (read gate, read up, write out)
AI = 4N / (6N) = 0.67 FLOP/byte → Memory-bound
```

---

## Bottleneck Analysis

### Decode Phase (B=1)

**Compute-bound kernels:**
- `cutlass_scaled_fp4_mm` (all GEMM projections)
- `bmm_fp8` (only at higher batch sizes)

**Memory-bound kernels:**
- `rmsnorm` / `fused_add_rmsnorm`
- `silu_and_mul`
- `topk_softmax` / `topk_sigmoid`
- `apply_rope_with_cos_sin_cache_inplace`

**Overall:** At B=1, GEMM kernels dominate but are compute-limited. Memory-bound kernels (norms, activations) contribute less to total time.

### Decode Phase (B=128)

At higher batch sizes, all kernels become more compute-bound:
- GEMM AI scales with batch size (more work per byte)
- BMM becomes compute-bound
- Memory-bound ops remain memory-bound but faster per-token

### Prefill Phase (B=1, S=1024)

**Compute-bound kernels:**
- All GEMM projections (large M = B×S)
- Attention (large sequence dimension)

**Memory-bound kernels:**
- Normalizations (smaller fraction of total time)
- Activations

**Overall:** Prefill is dominated by compute-bound operations due to large token counts.

---

## Scaling Behavior

### Batch Scaling (Decode)

As batch size B increases:
1. **GEMM AI stays roughly constant** (matrix dimensions unchanged for weight)
2. **BMM AI increases** (M grows with B)
3. **Memory-bound ops**: Stay memory-bound, but better amortization
4. **Throughput**: Improves until hitting compute ceiling

### Sequence Scaling (Prefill)

As sequence length S increases:
1. **GEMM**: Linear scaling in compute, remains compute-bound
2. **Attention**: Quadratic scaling (O(S²)), dominates at long sequences
3. **KV cache**: Linear memory growth

---

## Optimization Opportunities

### Compute-Bound Kernels
- Kernels at < 70% peak efficiency have room for improvement
- Check CUTLASS tile sizes, memory access patterns
- Consider occupancy tuning

### Memory-Bound Kernels
- Already at hardware bandwidth limits
- Fusion opportunities (e.g., fused_add_rmsnorm)
- Reduce memory traffic through kernel fusion

### Mixed Kernels (MLA Decode)
- At low batch: memory-bound (optimize bandwidth)
- At high batch: compute-bound (optimize FLOPS)
- Auto-tune based on batch size

---

## How to Generate Roofline Plot

```bash
cd deepseek_kernel_benchmarks/scripts

# Run benchmarks first
python bench_deepseek_kernels.py --phase all --output ../results/

# Generate roofline plot
python plot_roofline.py --input ../results/all_kernels.csv --output ../roofline.png --report ../roofline.md

# Generate separate plots per phase
python plot_roofline.py --input ../results/all_kernels.csv --output ../roofline.png --by-phase
```
