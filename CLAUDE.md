# SGLang DeepSeek-Only Project

## Project Goal
Shrink the SGLang repository (663K+ lines of code across 1945+ Python files) to a minimal codebase that:
- **ONLY** supports DeepSeek models (v2, v3, R1)
- Maintains **100% correctness** - no functionality changes
- Keeps **ALL optimizations** enabled for DeepSeek models
- Targets **NVIDIA GPU only** for multi-node deployment (remove CPU, NPU, AMD GPU support)
- **Preserves the same launch command and API** - server, API endpoints, and runtime infrastructure remain unchanged
- **Iterative removal approach** - only remove unused code, unreferenced code, and non-DeepSeek models

## Codebase Overview
- **Original:** ~663K lines → **Current:** ~291K lines (~56% reduction)
- **Repository Location:** `sglang/` (git submodule)
- **Main Source:** `sglang/python/sglang/`
- **Key Directories:**
  - `sglang/python/sglang/srt/` - Runtime system
  - `sglang/python/sglang/srt/models/` - Model implementations
  - `sglang/benchmark/deepseek_v3/` - DeepSeek benchmarks

## Target DeepSeek Model Architecture

### Primary Model: DeepSeek-R1 (NVIDIA FP4 v2)
- **Reference:** https://huggingface.co/nvidia/DeepSeek-R1-NVFP4-v2
- **Total Layers:** 61 layers (58 MoE + 3 standard transformer)
- **Key Features:** Multi-head Latent Attention (MLA), Mixture of Experts (MoE), NVIDIA FP4 quantization
- **Deployment Target:** Multi-node NVIDIA GPU cluster (GB200 Blackwell tested)

### DeepSeek Model Files
- `deepseek.py` - DeepSeek v1
- `deepseek_v2.py` - DeepSeek v2/v3/R1
- `deepseek_nextn.py` - NextN speculative decoding
- `deepseek_common/` - Shared components (MLA, MoE)

## Iterative Shrinking Strategy

**Core Principle:** This is an **iterative, dependency-driven** approach. We only remove code that is:
1. **Provably unused** by DeepSeek models
2. **Unreferenced** in the dependency graph
3. **Non-essential** for the API/server infrastructure

### Phase 1: Discovery & Dependency Analysis ✅
- Map DeepSeek dependencies (imports, layers, kernels, utilities)
- Identify hardware dependencies (CUDA kernels, NVIDIA optimizations)
- Build reference graph (keep list, remove list)

### Phase 2: Safe Removals ✅
- Remove non-DeepSeek models (~100+ model files)
- Remove hardware backends (NPU, AMD GPU, CPU-only)
- Remove non-DeepSeek benchmarks
- Validation after each batch (import checks, server launch)
- **Result:** ~40-60% code reduction

### Phase 3: Deep Cleanup ✅
- **Phase 3A:** Remove alternative architectures (Mamba, FLA, Vision) - 47 files, ~18,574 lines
- **Phase 3B:** Remove CPU/XPU/HPU/NPU platform support - 31 files, ~310 lines
- **Phase 3C:** Remove AMD/ROCm/HIP platform code - 110+ files, ~8,472 lines
- **Phase 3D:** Infrastructure cleanup (MindSpore, unused imports) - ~175 lines
- **Kernel Cleanup:** Remove unused sgl-kernel modules - 4 files, 282 lines
- **Result:** 208 files modified, ~27,578 lines removed

### Phase 4: Testing & Validation ✅ WORKING
- GB200 compilation and runtime testing
- Fix import errors and missing modules
- Benchmark validation (bench_one_batch, bench_serving)
- **Result:** First successful deployment on GB200

## Reduction Summary

| Metric | Value |
|--------|-------|
| Original Lines | ~663,000 |
| Current Lines | ~277,000 |
| Lines Removed | ~386,000 |
| Reduction | ~58% |
| Files Modified | 1,500+ |
| Models Removed | ~100+ |
| Platform Conditionals Removed | 0 NPU/HIP/XPU references remain |

## Current Code Attribution

### By Component (Total: ~277K lines)

| Component | Lines | % |
|-----------|------:|--:|
| **sglang/python/** | 177,328 | 64% |
| **sgl-kernel (CUDA/C++)** | 63,451 | 23% |
| **sgl-kernel (Python)** | 19,286 | 7% |
| **sgl-model-gateway** | 17,061 | 6% |

### sglang/python/sglang Breakdown

| Module | Lines | Description |
|--------|------:|-------------|
| srt/ | 165,116 | Runtime system |
| lang/ | 4,468 | Language frontend |

### srt/ Breakdown (165K lines)

| Module | Lines | Description |
|--------|------:|-------------|
| layers/ | 57,660 | Neural network layers |
| managers/ | 20,093 | Scheduler, workers, tokenizer |
| mem_cache/ | 13,990 | KV cache, radix cache, memory pool |
| entrypoints/ | 10,558 | HTTP server, OpenAI API |
| speculative/ | 8,678 | Speculative decoding |
| utils/ | 7,633 | Utilities |
| disaggregation/ | 6,340 | PD disaggregation |
| model_executor/ | 5,482 | Model execution |
| distributed/ | 5,323 | Tensor/pipeline parallelism |
| models/ | 4,527 | DeepSeek model implementations |
| model_loader/ | 3,874 | Weight loading |
| eplb/ | 3,117 | Expert load balancing |
| Other | 18,821 | compilation, function_call, sampling, etc. |

### layers/ Breakdown (58K lines)

| Module | Lines | Description |
|--------|------:|-------------|
| quantization/ | 18,646 | FP8, MXFP4, GPTQ, W8A8 |
| attention/ | 15,936 | MLA, FlashInfer, CutlassMLA, NSA |
| moe/ | 13,305 | Mixture of Experts |
| Core files | 8,411 | linear, rotary, logits, sampler |
| Other | 994 | utils, deep_gemm_wrapper |

## Registry & Analysis Tools

See `tools/` directory for:
- **`tools/registry.md`** - Authoritative kept/removed list
- **`tools/dead_code_analysis.md`** - Find NEW dead code
- **`tools/deletion_completion_analysis.md`** - Verify deleted items have no references

When user says:
- "run dead code analysis" → Follow `tools/dead_code_analysis.md`
- "run deletion completion analysis" → Follow `tools/deletion_completion_analysis.md`

## Removal Principles

1. **Check registry first** - See `tools/registry.md` for what's kept/removed
2. **Verify before delete** - grep for references before removal
3. **Test after removal** - Import checks and syntax verification
4. **Document in registry** - Update `tools/registry.md` after changes

## Current Status
- **Phase:** Phase 4 - Testing & Validation ✅ WORKING
- **Last Updated:** 2026-01-16
- **Lines of Code:** ~291,000 (From original ~663K, ~56% reduction)

## Working Milestones

### Milestone 2: AWQ/GPTQ/Marlin Removal (Current)

**Commit:** `d1acc3ac1` (2026-01-17)
**Branch:** `deepseek-only`

Removed unused quantization backends:
- AWQ: awq_triton.py, auto_round.py, sgl-kernel AWQ tests/benchmarks
- GPTQ: gptq.py, marlin_utils.py, marlin_utils_fp8.py, moe_wna16.py
- Compressed Tensors: entire directory (kept only utils.py for should_ignore_layer)
- Marlin MoE: moe_runner/marlin.py, fused_marlin_moe.py
- FP8 Marlin: removed fallback path
- Flex Attention: removed from attention_registry.py and server_args.py

**Tested on GB200:** WORKING ✅

---

### Milestone 1: Initial GB200 Deployment

**Commit:** `79db3fde9` (2026-01-16)
**Branch:** `deepseek-only`

First successful GB200 deployment:
- bench_one_batch mode: WORKING
- client-server mode: WORKING

This is the baseline working commit for DeepSeek-only deployment on GB200.


## What's Kept vs Removed (Summary)

See `tools/registry.md` for complete details.

## Key Decisions Made

### Text-Only R1
Keep ONLY text-only DeepSeek R1 models. Removed multimodal variants (deepseek_vl2.py, deepseek_janus_pro.py, deepseek_ocr.py).

### Quantization
- **Kept:** FP8, MXFP4 (primary for DeepSeek R1), GPTQ, W8A8, BitsAndBytes, ModelOpt
- **Removed:** AWQ (not needed for DeepSeek R1 FP4/FP8), GGUF (not compiled)

### NSA Module
Restored for future DeepSeek V3.2 support. FlashMLA methods are stubs (NotImplementedError).

## Safety Principles

1. Work in **small batches** (one directory at a time)
2. **Test after each removal** (no big bang approach)
3. **Maintain git history** for easy rollback
4. **Verify before delete:** Always check if code is referenced before removal
5. **Keep when uncertain:** If not sure if code is used, keep it

**Tools for Verification:**
- `grep -r "import.*<deleted_module>"` - Check for import references
- `python -c "import sglang.srt.models.deepseek_v2"` - Test imports
- `python -m py_compile <file>` - Syntax check

## Notes
- Original repo: https://github.com/sgl-project/sglang
- Workspace: `/Users/lpc/workspace/sglang-deepseek-only`
