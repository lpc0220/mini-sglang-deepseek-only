# Phase 3C: Files Modified

**Total Files Modified:** 59+ files
**Lines Removed:** 7,333+
**Platform Conditionals Removed:** 116+ (73 NPU + 32 HIP + 11 XPU)

## Files Modified by Category

### Distributed Communication (8 files)
- `distributed/parallel_state.py` - 1,716 lines removed
- `distributed/device_communicators/custom_all_reduce.py` - 407 lines removed
- `distributed/device_communicators/custom_all_reduce_ops.py` - 141 lines removed
- `distributed/device_communicators/custom_all_reduce_utils.py` - 79 lines removed
- `distributed/device_communicators/pymscclpp.py` - 2 lines removed
- `distributed/device_communicators/quick_all_reduce.py` - 1 line removed
- `distributed/device_communicators/torch_symm_mem.py` - 1 line removed

### Quantization Layers (7 files)
- `layers/quantization/fp8_kernel.py` - 1,368 lines removed
- `layers/quantization/awq.py` - 336 lines removed
- `layers/quantization/fp8.py` - 1 line removed (+ variable assignments)
- `layers/quantization/fp8_utils.py` - Fixed HIP conditional
- `layers/quantization/gguf.py` - Fixed HIP conditional
- `layers/quantization/mxfp4.py` - 13 lines removed
- `layers/quantization/quark/schemes/quark_w4a4_mxfp4.py` - 11 lines removed
- `layers/quantization/petit.py` - Fixed HIP compatibility check

### MoE Infrastructure (4 files)
- `layers/moe/ep_moe/layer.py` - 480 lines removed
- `layers/moe/moe_runner/deep_gemm.py` - 2 lines removed
- `layers/moe/token_dispatcher/deepep.py` - 2 lines removed
- `layers/moe/topk.py` - 3 lines removed

### Core Layers (10 files)
- `layers/linear.py` - 405 lines removed
- `layers/layernorm.py` - 5 lines removed
- `layers/activation.py` - 1 line removed
- `layers/communicator.py` - 7 lines removed
- `layers/rotary_embedding.py` - 4 lines removed
- `layers/vocab_parallel_embedding.py` - 2 lines removed
- `layers/elementwise.py` - Fixed HIP ternary operators
- `layers/dp_attention.py` - 2 lines removed
- `layers/logits_processor.py` - 2 lines removed
- `layers/utils/multi_platform.py` - 8 lines removed

### Attention Backends (2 files)
- `layers/attention/nsa_backend.py` - 13 lines removed
- `layers/attention/triton_ops/decode_attention.py` - Fixed HIP conditional

### Model Execution (4 files)
- `model_executor/model_runner.py` - 36 lines removed
- `model_executor/model_runner_kv_cache_mixin.py` - 5 lines removed
- `model_executor/forward_batch_info.py` - 2 lines removed
- `model_executor/cuda_graph_runner.py` - 3 lines removed

### Model Loading & Memory (3 files)
- `model_loader/loader.py` - 1,126 lines removed
- `mem_cache/memory_pool.py` - 803 lines removed
- `mem_cache/memory_pool_host.py` - 7 lines removed

### Managers & Profiling (3 files)
- `managers/scheduler_profiler_mixin.py` - 15 lines removed
- `managers/mm_utils.py` - 229 lines removed
- `utils/profile_utils.py` - 15 lines removed

### Speculative Execution (5 files)
- `speculative/eagle_info_v2.py` - 19 lines removed
- `speculative/eagle_utils.py` - 34 lines removed
- `speculative/eagle_worker.py` - 3 lines removed
- `speculative/eagle_worker_v2.py` - 4 lines removed
- `speculative/multi_layer_eagle_worker.py` - 2 lines removed
- `speculative/spec_utils.py` - 5 lines removed

### DeepSeek Models (1 file)
- `models/deepseek_nextn.py` - 1 line removed

### Other Components (5 files)
- `batch_overlap/two_batch_overlap.py` - 2 lines removed (+ ternary fix)
- `constrained/xgrammar_backend.py` - 8 lines removed
- `multimodal/processors/base_processor.py` - 2 lines removed

## Patterns Removed

### NPU Patterns (73+ occurrences)
- `if _is_npu:` blocks
- `elif _is_npu:` branches
- `if _is_cuda or _is_npu:` → `if _is_cuda:`
- `if not _is_npu:` (condition removed)
- `@torch.compile(disable=_is_npu)` → `disable=False`
- `_is_npu = is_npu()` variable assignments
- NPU-specific imports

### HIP Patterns (32+ occurrences)
- `if _is_hip:` blocks
- `elif _is_hip:` branches
- `if _is_cuda or _is_hip:` → `if _is_cuda:`
- `max_warps = 16 if _is_hip else 32` → `max_warps = 32`
- `_use_hip_int4 = ... and _is_hip` → `_use_hip_int4 = False`
- `IS_CUSTOM_AR_AVAILABLE = _is_cuda or _is_hip` → `_is_cuda`
- `IS_QUICK_AR_AVAILABLE = _is_hip` → `False`
- `dynamic=_is_hip and ...` → `dynamic=False`
- `_use_aiter` variable assignments

### XPU Patterns (11 occurrences)
- `_is_xpu = is_xpu()` variable assignments
- `if _is_cuda or _is_xpu:` → `if _is_cuda:`
- `and not (_is_xpu)` (removed)
- `if not (_is_npu or _is_xpu):` (condition removed)

## Verification

All 59+ files verified clean:
- ✅ No `_is_npu` references
- ✅ No `_is_hip` references
- ✅ No `_is_xpu` references
- ✅ CUDA references preserved (144 total)

## Tools Used

1. `phase3c_complete_cleanup.py` - Automated pattern matching (51 files)
2. `phase3c_final_cleanup.py` - Edge case handling (13 files)
3. `phase3c_ultra_final_cleanup.py` - Complex patterns (19 files)
4. Manual edits - 5 complex multiline patterns
5. `phase3c_remove_xpu.py` - XPU cleanup (8 files)

---

**Phase 3C Complete:** All platform-specific conditionals removed, SGLang is now 100% NVIDIA CUDA-only.
