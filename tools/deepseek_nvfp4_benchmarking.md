# DeepSeek Kernel Benchmark Script Generator

When user says "do deepseek kernel benchmarking", follow these steps to **generate benchmark scripts** for profiling kernel performance in DeepSeek-R1-NVFP4-v2 execution.

**Purpose:** Generate Python scripts that measure kernel performance (latency, FLOPS, bandwidth) across batch sizes and sequence lengths, comparing against hardware peak. This tool generates and verifies scripts - it does NOT run the benchmarks.

**Output:**
- `deepseek_kernel_benchmarks/scripts/bench_{kernel}.py` - Individual benchmark scripts (one per kernel)
- `deepseek_kernel_benchmarks/scripts/run_all_benchmarks.py` - Orchestrator script
- `deepseek_kernel_benchmarks/scripts/plot_roofline.py` - Visualization script

**Input:** `results/deepseek_nvfp4_ops.md` - Execution path with op shapes and kernel mappings

---

## Step 1: Read All Kernels from deepseek_nvfp4_ops.md

Read `results/deepseek_nvfp4_ops.md` and extract ALL kernels from the **"## Kernel Summary"** section at the end of the file. This section contains the authoritative list of kernels to benchmark.

The kernel summary table has this format:
```
| Kernel | Source | Ops Using It |
|--------|--------|--------------|
| `rmsnorm` | sgl-kernel | ... |
| `fused_add_rmsnorm` | sgl-kernel | ... |
...
```

**Important:** Every kernel listed in that table must have a corresponding benchmark file.

---

## Step 2: Verify or Generate Benchmark Files

For each kernel from the summary, check if a benchmark file exists:

```bash
ls deepseek_kernel_benchmarks/scripts/bench_{kernel_name}.py
```

**If benchmark file exists:** Verify it benchmarks the correct kernel with proper shapes from `deepseek_nvfp4_ops.md`.

**If benchmark file is missing:** Generate a new benchmark script as described below.

### How to Generate a Kernel Benchmark

For each missing kernel, create `bench_{kernel_name}.py` that:

1. **Imports the kernel** from its source:
   - `sgl-kernel`: `from sgl_kernel import {kernel_name}`
   - `flashinfer`: `from flashinfer import {kernel_name}` or `from flashinfer.triton.moe import ...`
   - `triton`: `from sglang.srt.layers.moe.fused_moe_triton import fused_moe`

2. **Creates input tensors** with shapes from `deepseek_nvfp4_ops.md`:
   - Look up the kernel's ops in the ops file to find tensor shapes
   - Use model parameters (H, Nh, E, K, I, etc.) for dimensions
   - Total tokens = B (decode) or B × S (prefill)

3. **Wraps the kernel call** in a function for timing:
   - The wrapper should call the kernel with all required arguments
   - Handle any output tensors or in-place operations

4. **Measures latency** using CUDA graph benchmarking:
   - Use `triton.testing.do_bench_cudagraph()` for accurate GPU timing
   - Warmup iterations, then timed runs, report median

5. **Computes metrics** based on kernel type:
   - **Compute-bound** (GEMM, BMM): Calculate FLOPS, compare to peak TFLOPS
   - **Memory-bound** (Norm, RoPE, Activation): Calculate bytes transferred, compare to peak bandwidth
   - **Mixed** (Attention, MoE): Calculate both, determine which is limiting

6. **Returns a BenchmarkResult** with all fields populated

7. **Implements `run_benchmarks(batch_sizes, seq_lens, output_dir)`** that:
   - Iterates over decode phase (all B values, S=1)
   - Iterates over prefill phase (B=1,2,4,8 × all S values)
   - Saves results to CSV

### File Structure

```
deepseek_kernel_benchmarks/
├── scripts/
│   ├── bench_{kernel_name}.py  # One per kernel
│   ├── run_all_benchmarks.py   # Orchestrator
│   └── plot_roofline.py        # Visualization
└── results/
    ├── {kernel_name}.csv       # Per-kernel results
    ├── all_kernels.csv         # Aggregated
    └── benchmark_summary.md    # Run status
```

---

## Benchmarking Methodology

### Benchmark Parameters

Each kernel is benchmarked across a sweep of batch sizes (B) and sequence lengths (S):

| Parameter | Default Values | Description |
|-----------|----------------|-------------|
| Batch sizes (B) | 1, 2, 4, 8, 16, 32, 64, 128 | Number of requests |
| Sequence lengths (S) | 128, 256, 512, 1024, 2048 | Tokens per request |

### Benchmark Phases

**Decode Phase (S=1):**
- Iterates over all batch sizes: B = 1, 2, 4, 8, 16, 32, 64, 128
- Total tokens = B (one token per request)
- Simulates autoregressive token generation

**Prefill Phase:**
- Iterates over batch sizes B = 1, 2, 4, 8 (first 4 values)
- For each B, iterates over all sequence lengths: S = 128, 256, 512, 1024, 2048
- Total tokens = B × S
- Simulates prompt processing

### Timing Method

- Uses CUDA graph capture to eliminate CPU overhead
- Warmup: 10 iterations (discarded)
- Timed runs: 100 iterations
- Reports median latency in milliseconds

### Output Results

Each benchmark produces a result with these fields, saved as CSV:

| Field | Description |
|-------|-------------|
| `kernel` | Kernel name (e.g., `rmsnorm`) |
| `op` | Operation name (e.g., `input_layernorm`) |
| `phase` | `decode` or `prefill` |
| `B` | Batch size |
| `S` | Sequence length |
| `M, N, K` | Matrix dimensions (for GEMM/BMM) |
| `latency_ms` | Median latency in milliseconds |
| `gflops` | Achieved GFLOPS (compute-bound) |
| `bandwidth_gbs` | Achieved bandwidth GB/s (memory-bound) |
| `peak_pct` | Percentage of hardware peak |
| `arith_intensity` | Arithmetic intensity (FLOP/byte) |
| `bound` | `compute` or `memory` |

**CSV Format Example:**
```csv
kernel,op,phase,B,S,M,N,K,latency_ms,gflops,peak_pct,bandwidth_gbs,arith_intensity,bound
cutlass_scaled_fp4_mm,q_b_proj,decode,1,1,1,24576,1536,0.012300,1234.50,13.72,,7200.00,compute
rmsnorm,input_layernorm,decode,1,1,1,7168,0,0.003000,,80.00,6400.00,1.25,memory
```

**Output Files:**
- `{kernel_name}.csv` - Per-kernel results
- `all_kernels.csv` - Aggregated from all kernel CSVs
- `benchmark_summary.md` - Run status and statistics

### FLOPS/Bandwidth Formulas by Kernel Type

| Kernel Type | FLOPS Formula | Bytes Formula |
|-------------|---------------|---------------|
| GEMM | `2 * M * N * K` | `M*K + K*N + M*N` (× dtype_size) |
| BMM | `2 * batch * M * N * K` | `batch * (M*K + K*N + M*N)` |
| RMSNorm | `5 * N` (approx) | `2 * N * dtype_size` (read + write) |
| Fused Add RMSNorm | `6 * N` | `4 * N * dtype_size` (read x, residual; write x, residual) |
| SiLU and Mul | `4 * N` | `3 * N * dtype_size` (read gate, up; write out) |
| RoPE | `4 * tokens * heads * head_dim` | `4 * tokens * heads * head_dim * dtype_size` |
| TopK | `tokens * num_experts * 5` | `tokens * num_experts * 4 + tokens * topk * 8` |
| Attention | `4 * B * Nh * seq_len * head_dim` | Q + KV cache + output bytes |

---

## Step 3: Verify or Generate Orchestration Scripts

Check if orchestration scripts exist in `deepseek_kernel_benchmarks/scripts/`:

- `run_all_benchmarks.py` - Runs all kernel benchmarks and aggregates results
- `plot_roofline.py` - Generates roofline visualization from results

**If missing:** Generate these scripts. The orchestrator should:
- Import and run each `bench_{kernel_name}.py` module
- Support `--list` to show available kernels
- Support `--kernels` to run specific kernels only
- Support `--batch-sizes` and `--seq-lens` to customize sweep parameters
- Aggregate all per-kernel CSVs into `all_kernels.csv`
- Generate `benchmark_summary.md` with run status

---

## Step 4: Verify Generated Scripts (CRITICAL)

**This step is mandatory.** After generating any benchmark script, perform thorough verification before considering it complete. Do not skip any checks.

### 4.1 Syntax Verification

Run syntax check on every generated file:
```bash
python -m py_compile bench_{kernel_name}.py
```
- Must pass with no errors
- Fix any syntax issues immediately

### 4.2 Import Verification

Verify all imports are correct:
- **Kernel import path** - Is the kernel imported from the correct module?
  - Cross-check against `deepseek_nvfp4_ops.md` "Source" column
  - Verify the exact function name exists in that module
- **Dependency imports** - Are all required modules imported (torch, triton, etc.)?
- **Utility imports** - Are shared constants and functions imported correctly?

### 4.3 Shape Verification (CRITICAL)

**Carefully** compare tensor shapes against `deepseek_nvfp4_ops.md`:

- **Input tensor shapes** - Do they match the op's input shapes exactly?
- **Weight tensor shapes** - Are dimensions in the correct order (e.g., `[out_features, in_features]` vs `[in_features, out_features]`)?
- **Output tensor shapes** - Are output buffers sized correctly?
- **Model parameters** - Are H, Nh, E, K, I, Lq, Lkv, etc. used correctly?
- **Token calculation** - Is `tokens = B` for decode and `tokens = B * S` for prefill?
- **Dtype** - Are tensors created with correct dtype (bfloat16, float8, etc.)?

### 4.4 Kernel Call Verification

Verify the kernel is called correctly:
- **Argument order** - Do arguments match the kernel's function signature?
- **Required vs optional args** - Are all required arguments provided?
- **In-place operations** - Are output tensors handled correctly?
- **Device placement** - Are all tensors on CUDA?

### 4.5 Metric Calculation Verification

Verify FLOPS and bandwidth calculations:
- **Kernel category** - Is it correctly classified as compute-bound, memory-bound, or mixed?
- **FLOPS formula** - Does it match the formula in the "FLOPS/Bandwidth Formulas" table?
- **Bytes formula** - Does it account for all input AND output memory transfers?
- **Dtype sizes** - Are byte calculations using correct dtype sizes (bf16=2, fp8=1, fp4=0.5)?
- **Peak comparison** - Is peak_pct comparing against the correct hardware peak (FP4, FP8, or FP16)?

### 4.6 Iteration Logic Verification

Verify the benchmark loop:
- **Decode phase** - Iterates over all B values with S=1?
- **Prefill phase** - Iterates over B=1,2,4,8 × all S values?
- **Results collection** - Are all results appended and saved?

### 4.7 Final Checklist

Before marking a script as complete:
- [ ] Syntax check passes
- [ ] All imports resolve correctly
- [ ] Tensor shapes match ops file exactly
- [ ] Kernel call matches function signature
- [ ] FLOPS/bandwidth formulas are correct for kernel type
- [ ] Decode and prefill iterations are correct
- [ ] CSV output fields are all populated

**Do not proceed until all checks pass. Fix any issues found and re-verify.**

---

## Model and Hardware Constants

When generating benchmark scripts, use these constants:

### DeepSeek-R1 Model Parameters

| Symbol | Value | Parameter |
|--------|-------|-----------|
| H | 7168 | hidden_size |
| Nh | 128 | num_heads |
| Lq | 1536 | q_lora_rank |
| Lkv | 512 | kv_lora_rank |
| Dn | 128 | qk_nope_head_dim |
| Dr | 64 | qk_rope_head_dim |
| Dv | 128 | v_head_dim |
| Dq | 192 | qk_head_dim (Dn + Dr) |
| E | 256 | n_routed_experts |
| K | 8 | num_experts_per_tok |
| I | 2048 | moe_intermediate_size |

### GB200 Hardware Specifications

| Metric | Value |
|--------|-------|
| Peak FP4 Compute | 9000 TFLOPS |
| Peak FP8 Compute | 4500 TFLOPS |
| Peak FP16/BF16 Compute | 2250 TFLOPS |
| Peak HBM Bandwidth | 8000 GB/s |

---

## Notes

- **Read kernels from `results/deepseek_nvfp4_ops.md`** - The "## Kernel Summary" section is the authoritative source
- **One benchmark file per kernel** - Verify with `python run_all_benchmarks.py --list`
- **Internal kernels**: Some kernels may report "not available" if APIs are internal
- **Prefill vs Decode**: Different shapes, different bottlenecks
- **FP4 quantization**: Most GEMMs use FP4 weights with FP8/FP16 compute
- **MoE complexity**: Expert kernels have irregular memory access patterns
