# Phase 3C: AMD/ROCm/HIP Cleanup - Progress Report

**Date:** 2026-01-11
**Status:** Partial Completion (MoE Layer Priority Files Cleaned)

## Executive Summary

Phase 3C successfully removed AMD/ROCm/HIP conditional branches from 4 critical MoE layer files that are essential for DeepSeek model support. All quantization implementations were preserved, with only platform-specific conditional branches removed.

## Files Successfully Cleaned

### Priority 1: MoE Layers (Critical for DeepSeek) ✅

| File | Lines Removed | Status | Changes Made |
|------|--------------|--------|--------------|
| `fused_moe_triton_kernels.py` | ~10 | ✅ Complete | Removed `_is_hip`, `_use_aiter`, simplified quantization conditionals |
| `fused_moe_triton/fused_moe.py` | ~50 | ✅ Complete | Removed HIP import branches, aiter logic, vllm fallbacks |
| `fused_moe_triton/layer.py` | ~15 | ✅ Complete | Removed HIP scale adjustments, aiter expert masking |
| `router.py` | ~3 | ✅ Complete | Simplified max_warps (32 vs 16 for HIP) |

**Total:** 4 files cleaned, ~78 lines removed

### Syntax Validation ✅
All cleaned files pass Python syntax validation:
```bash
python3 -m py_compile sglang/python/sglang/srt/layers/moe/router.py  # ✅ Pass
python3 -m py_compile sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py  # ✅ Pass
python3 -m py_compile sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py  # ✅ Pass
python3 -m py_compile sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py  # ✅ Pass
```

## Cleanup Patterns Applied

### Pattern 1: Remove Module-Level Variables
```python
# BEFORE:
_is_hip = is_hip()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# AFTER:
_is_cuda = is_cuda()
```

### Pattern 2: Remove HIP Import Branches
```python
# BEFORE:
if _is_cuda:
    from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul
    if _use_aiter:
        from aiter import moe_sum

# AFTER:
from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul
```

### Pattern 3: Simplify OR Conditions
```python
# BEFORE:
if _is_cuda or _is_hip:
    code()

# AFTER:
if _is_cuda:
    code()
```

### Pattern 4: Remove elif HIP Branches
```python
# BEFORE:
if _is_cuda:
    A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
else:
    A, A_scale = per_token_group_quant_fp8(A, block_k)

# AFTER:
A, A_scale = sglang_per_token_group_quant_fp8(A, block_k)
```

### Pattern 5: Remove HIP-Specific Adjustments
```python
# BEFORE:
if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
    loaded_weight = loaded_weight * 2.0  # AMD e4m3fnuz adjustment

# AFTER:
# (removed entirely - CUDA only)
```

## Remaining Files (31 files with HIP references)

### MoE Layer Files (8 remaining)
- `layers/moe/ep_moe/layer.py` - Expert parallelism MoE
- `layers/moe/moe_runner/triton.py` - Triton MoE runner
- `layers/moe/moe_runner/deep_gemm.py` - DeepGEMM wrapper
- `layers/moe/fused_moe_triton/moe_align_block_size.py` - Block alignment
- `layers/moe/fused_moe_triton/fused_moe_triton_config.py` - Triton config
- `layers/moe/token_dispatcher/standard.py` - Standard dispatcher
- `layers/moe/token_dispatcher/deepep.py` - DeepEP dispatcher
- `layers/moe/topk.py` - TopK selection (previously cleaned, verify)

### Quantization Files (14 remaining)
**CRITICAL:** Only remove HIP branches, preserve ALL quantization implementations!

- `layers/quantization/fp8.py` - FP8 quantization (PRESERVE implementation!)
- `layers/quantization/fp8_kernel.py` - FP8 kernels
- `layers/quantization/fp8_utils.py` - FP8 utilities
- `layers/quantization/mxfp4.py` - MXFP4 quantization (DeepSeek R1 uses this!)
- `layers/quantization/awq.py` - AWQ quantization
- `layers/quantization/gguf.py` - GGUF quantization
- `layers/quantization/unquant.py` - Unquantized method
- `layers/quantization/__init__.py` - Quant module init
- `layers/quantization/petit.py` - Petit quantization
- `layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`
- `layers/quantization/quark/quark_moe.py`
- `layers/quantization/quark/schemes/quark_w8a8_fp8.py`
- `layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`

### Attention Files (8 remaining)
- `layers/attention/nsa_backend.py` - NSA backend
- `layers/attention/nsa/nsa_indexer.py` - NSA indexer
- `layers/attention/nsa/tilelang_kernel.py` - TileLang kernel
- `layers/attention/triton_ops/double_sparsity_attention.py`
- `layers/attention/triton_ops/decode_attention.py`
- `layers/attention/triton_ops/extend_attention.py`
- `layers/attention/triton_ops/prefill_attention.py`
- `layers/attention/merge_state.py`

### Server Configuration (1 remaining)
- `srt/server_args.py` - Remove ROCm device arguments

## Critical Warnings

### ⚠️ Quantization Files - DO NOT REMOVE Implementations!
When cleaning quantization files:
- ✅ **REMOVE:** `elif _is_hip:` branches that dispatch to ROCm kernels
- ✅ **REMOVE:** AMD-specific optimizations (aiter library)
- ✅ **REMOVE:** HIP device detection and platform checks
- ❌ **PRESERVE:** All quantization algorithms (FP8, MXFP4, AWQ, GPTQ, etc.)
- ❌ **PRESERVE:** CUDA kernel implementations
- ❌ **PRESERVE:** Quantization configuration classes

**Example (fp8.py):**
```python
# BEFORE (lines ~350):
if _is_cuda:
    result = cuda_fp8_kernel(...)  # ✅ KEEP THIS
elif _is_hip:
    result = hip_fp8_kernel(...)   # ❌ REMOVE THIS BRANCH ONLY

# AFTER:
if _is_cuda:
    result = cuda_fp8_kernel(...)  # ✅ KEPT
```

### ⚠️ DeepSeek R1 Dependency: MXFP4 Quantization
DeepSeek-R1 (NVIDIA FP4 v2) uses MXFP4 quantization. The following files MUST preserve CUDA MXFP4 implementations:
- `layers/quantization/mxfp4.py`
- `layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`
- `layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`

## Next Steps

### Immediate (Complete Phase 3C)
1. Clean remaining 8 MoE layer files
2. **CAREFULLY** clean 14 quantization files (preserve implementations!)
3. Clean 8 attention layer files
4. Clean `server_args.py`
5. Run full syntax validation
6. Generate final statistics

### Testing (Phase 4)
After cleanup:
1. Verify imports resolve correctly
2. Test single-layer DeepSeek model construction
3. Validate MoE routing logic
4. Test MXFP4 quantization utilities
5. Ensure no broken CUDA kernel references

## Commands for Remaining Cleanup

### Find All HIP References
```bash
grep -r "_is_hip\|is_hip()\|_use_aiter" sglang/python/sglang/srt/layers/ \
    --include="*.py" | grep -v "__pycache__"
```

### Validate Syntax After Cleanup
```bash
find sglang/python/sglang/srt/layers -name "*.py" \
    -exec python3 -m py_compile {} \;
```

### Count Lines Removed
```bash
git diff --stat origin/main | tail -1
```

## Success Criteria

- ✅ All MoE files cleaned (4/12 complete)
- ⏳ All quantization implementations preserved (not started)
- ⏳ All attention files cleaned (not started)
- ⏳ Server args cleaned (not started)
- ✅ No syntax errors (4/4 files validated)
- ⏳ Import graph integrity verified (pending)

## Estimated Remaining Work

- **Files to clean:** 31 files
- **Estimated lines to remove:** ~200-300 lines
- **Risk level:** Medium (quantization files require careful review)
- **Time estimate:** 2-3 hours for careful manual cleanup

## Repository Status

- **Current branch:** main
- **Total lines of code:** 663,394 (baseline)
- **Lines removed in Phase 3C:** ~78 (0.01%)
- **Target reduction:** 70-80% (Phase 3 complete)

---

**Report generated:** 2026-01-11
**Phase:** 3C (AMD/ROCm/HIP Cleanup) - In Progress
**Next milestone:** Complete remaining 31 files, validate all changes
