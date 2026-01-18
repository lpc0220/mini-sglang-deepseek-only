# DeepSeek Kernel Tracing Tool

When user says "do deepseek kernel tracing", follow these steps to trace all kernel calls from DeepSeek models back to their source definitions.

**Output:** `tools/kernel_traces.md` - A file containing all kernel call chains from use to definition.

---

## Step 1: Identify DeepSeek Model Entry Points

Start from the DeepSeek model files:
```
sglang/python/sglang/srt/models/deepseek.py
sglang/python/sglang/srt/models/deepseek_v2.py
sglang/python/sglang/srt/models/deepseek_nextn.py
sglang/python/sglang/srt/models/deepseek_common/
```

---

## Step 2: Identify Kernel Sources

Kernels can come from these locations:

### 2.1 sgl-kernel (CUDA kernels)
```bash
# Python bindings
ls sglang/sgl-kernel/python/sgl_kernel/*.py

# Key modules: attention, moe, gemm, quantization, sampling, etc.
```

### 2.2 FlashInfer
```bash
# External library, imported as:
grep -rn "from flashinfer\|import flashinfer" --include="*.py" sglang/python/
```

### 2.3 Triton JIT Kernels
```bash
# In-repo Triton kernels
ls sglang/python/sglang/srt/layers/moe/fused_moe_triton/
ls sglang/python/sglang/srt/layers/quantization/*kernel*.py

# Look for @triton.jit decorators
grep -rn "@triton.jit" --include="*.py" sglang/python/
```

### 2.4 torch.ops / Custom Ops
```bash
# Custom CUDA ops registered via torch
grep -rn "torch.ops\." --include="*.py" sglang/python/
```

---

## Step 3: Trace Call Chains

For each kernel found, trace backwards from usage to definition:

### 3.1 MLA (Multi-head Latent Attention)
```bash
# Find MLA kernel usage in DeepSeek models
grep -rn "forward_absorb\|forward_decode\|forward_extend" sglang/python/sglang/srt/models/deepseek*.py
grep -rn "MLA\|mla" sglang/python/sglang/srt/layers/attention/

# Trace to backend implementations
grep -rn "class.*Backend" sglang/python/sglang/srt/layers/attention/
```

### 3.2 MoE (Mixture of Experts)
```bash
# Find MoE kernel usage
grep -rn "fused_moe\|FusedMoE\|MoeRunner" sglang/python/sglang/srt/models/deepseek*.py
grep -rn "fused_moe" sglang/python/sglang/srt/layers/moe/

# Trace to Triton kernels
grep -rn "fused_moe_kernel\|invoke_fused_moe" sglang/python/sglang/srt/layers/moe/
```

### 3.3 Linear/GEMM Operations
```bash
# Find linear layer usage
grep -rn "ColumnParallelLinear\|RowParallelLinear\|QKVParallelLinear" sglang/python/sglang/srt/models/deepseek*.py

# Trace to quantized implementations
grep -rn "apply.*linear\|gemm" sglang/python/sglang/srt/layers/quantization/
```

### 3.4 Rotary Embeddings
```bash
grep -rn "rotary\|RotaryEmbedding" sglang/python/sglang/srt/models/deepseek*.py
grep -rn "apply_rotary" sglang/python/sglang/srt/layers/
```

### 3.5 Sampling Kernels
```bash
grep -rn "top_k\|top_p\|sample" sglang/python/sglang/srt/layers/sampler.py
grep -rn "sampling" sglang/sgl-kernel/python/sgl_kernel/
```

---

## Step 4: Build Call Chain Format

For each kernel, document the call chain in this format:

```
### [Kernel Category]: [Kernel Name]

**Call Chain:**
```
[DeepSeek Model File]:[Line] [Function/Class]
  └─> [Intermediate Layer]:[Line] [Function/Class]
      └─> [Kernel Wrapper]:[Line] [Function/Class]
          └─> [Kernel Definition]:[Line] [Kernel Function]
```

**Source:** [sgl-kernel | flashinfer | triton | torch.ops]
**File:** [path to kernel definition]
```

---

## Step 5: Generate Output File

Create `tools/kernel_traces.md` with:

1. **Summary Table** - List of all kernels with their source
2. **Detailed Traces** - Full call chain for each kernel
3. **Kernel Source Breakdown** - Grouped by source (sgl-kernel, flashinfer, triton, etc.)

### Output Template:

```markdown
# DeepSeek Kernel Traces

Generated: [DATE]
DeepSeek Models: deepseek.py, deepseek_v2.py, deepseek_nextn.py

## Summary

| Kernel | Category | Source | Definition File |
|--------|----------|--------|-----------------|
| ... | ... | ... | ... |

## Detailed Call Chains

### Attention Kernels
[traces...]

### MoE Kernels
[traces...]

### Quantization Kernels
[traces...]

### Sampling Kernels
[traces...]

### Other Kernels
[traces...]

## Kernel Source Breakdown

### sgl-kernel
- [list of kernels]

### FlashInfer
- [list of kernels]

### Triton JIT
- [list of kernels]

### torch.ops
- [list of kernels]
```

---

## Step 6: Verification

After generating the trace file:

```bash
# Verify all referenced files exist
grep -o "sglang/[^:]*" tools/kernel_traces.md | sort -u | while read f; do
  test -f "$f" || echo "MISSING: $f"
done
```

---

## Quick Commands

```bash
# Find all kernel-like function calls in DeepSeek models
grep -rn "kernel\|_kernel\|triton\|cuda\|sgl_kernel\|flashinfer" --include="*.py" sglang/python/sglang/srt/models/deepseek*.py

# Find all imports in DeepSeek models
grep -rn "^from\|^import" sglang/python/sglang/srt/models/deepseek*.py | grep -v __pycache__

# Find sgl_kernel function calls
grep -rn "sgl_kernel\." --include="*.py" sglang/python/sglang/srt/
```

---

## Notes

- Focus on **compute-intensive** kernels (attention, MoE, GEMM, sampling)
- Include both **forward** and **backward** paths if applicable
- Note any **conditional paths** (e.g., different backends based on config)
- Track **quantization variants** (FP8, INT8, etc.)
