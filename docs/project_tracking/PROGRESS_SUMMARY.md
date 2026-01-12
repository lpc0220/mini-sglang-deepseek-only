# SGLang DeepSeek-Only Project - Progress Summary

**Last Updated:** 2026-01-11
**Phase:** Phase 2 - Safe Removals (In Progress)

---

## 📊 Overall Statistics

### Total Removed:
- **Files:** 765 files
- **Lines:** ~205,812 lines deleted
- **Git Commits:** 10 batches completed

### Breakdown by Batch:

| Batch | Category | Files | Lines | Commit |
|-------|----------|-------|-------|--------|
| 1 | DeepSeek Multimodal | 8 | ~5,118 | 6ee1d9cbf |
| 2a-2c | Non-DeepSeek Models | 117 | ~60,259 | 415da9e61, f9e6b646a, 1afecec6d |
| 3 | NPU/Ascend Backends | 32 | ~12,648 | 99a18e122 |
| 4 | Configs & Processors | 47 | ~8,538 | e2b0f0260 |
| 5 | Non-DeepSeek Benchmarks | 156 | ~26,051 | 8daf6e213 |
| 6 | Test Files & AMD/Ascend | 63 | ~6,280 | a04c2e2de |
| 7 | CPU/Intel Backends | 27 | ~5,781 | 032c04030 |
| 8 | Hardware Tests & MM Examples | 13 | ~1,625 | c2545598b |
| 9 | Multimodal Gen & Docs | 249 | ~78,459 | 41d0cd9c7 |
| 10 | Remaining Platform/Model Docs | 7 | ~1,053 | 467cc59e8 |
| **TOTAL** | | **765** | **~205,812** | |

---

## ✅ Completed Work

### Batch 1: DeepSeek Multimodal Variants
**Removed:** 8 files (~5,118 lines)
- 3 multimodal model files (janus_pro, ocr, vl2)
- 2 config files
- 3 processor files
- Cleaned up imports and utility functions

### Batch 2: Non-DeepSeek Models (2a-2c)
**Removed:** 117 model files (~60,259 lines)
- **Batch 2a (A-G):** 37 files - Apertus through Grok
- **Batch 2b (H-Q):** 66 files - Hunyuan through Qwen
- **Batch 2c (R-Z):** 14 files - Radio through Yivl

### Batch 3: Hardware Backends
**Removed:** 32 files (~12,648 lines)
- 17 NPU backend files
- 3 Ascend disaggregation files
- 3 Ascend test files
- 9 additional model stragglers

### Batch 4: Orphaned Configs & Processors
**Removed:** 47 files (~8,538 lines)
- 23 config files for deleted models
- 22 multimodal processor files
- Cleaned configs/__init__.py

### Batch 5: Non-DeepSeek Benchmarks
**Removed:** 156 files (~26,051 lines)
- 36 benchmark directories
- Kept: `benchmark/deepseek_v3/` only

### Batch 6: Test Files and AMD/Ascend Infrastructure
**Removed:** 63 files (~6,280 lines)
- 21 test files for removed models
- 29 Ascend NPU test files (entire directory)
- 8 AMD GPU test files (entire directory, includes DeepSeek AMD benchmarks)
- 5 platform documentation files

**Note:** DeepSeek AMD tests (v3, v31, R1) were removed as AMD-specific infrastructure. We're targeting NVIDIA GPU only.

### Batch 7: CPU and Intel Hardware Backends
**Removed:** 27 files (~5,781 lines)
- 4 CPU-specific implementation files (cpu_monitor, intel_amx_backend, amx_utils, cpu_graph_runner)
- 3 Intel XPU/HPU backend files (xpu_backend, xpu_communicator, hpu_communicator)
- 20 CPU test files (test/srt/cpu/ and test/manual/cpu/)
- Code cleanup in attention_registry, managers, and model_runner

**Note:** Intel AMX (Advanced Matrix Extensions), XPU (Arc GPU), and HPU (Habana) backends removed. NVIDIA GPU only.

### Batch 8: Hardware Tests and Multimodal Examples
**Removed:** 13 files (~1,625 lines)
- 3 hardware-specific test files (XPU, ROCm)
- 2 Qwen test files (LoRA, stress test)
- 4 multimodal example files (runtime/multimodal/)
- 2 LLaVA video example files
- 1 quick start example (llava_next)
- 1 vision template (sarashina_vl)

**Note:** Cleanup of remaining hardware tests and multimodal examples after model removal.

### Batch 9: Multimodal Generation Module and Documentation
**Removed:** 249 files (~78,459 lines)
- **python/sglang/multimodal_gen/** - Entire diffusion/image generation module (240 files)
  - Models: Wan, FastWan, Hunyuan, Qwen-Image, Flux, ZImage video/image generation
  - Infrastructure: VAEs, encoders, schedulers, pipelines, attention backends
  - Platforms: AMD ROCm, CPU, CUDA support for diffusion models
  - Runtime: Distributed inference, HTTP server, OpenAI API compatibility
  - Tests and benchmarks for diffusion models
- **Documentation (9 files):**
  - docs/basic_usage/llama4.md
  - docs/basic_usage/qwen3.md, qwen3_vl.md
  - docs/platforms/ascend_npu*.md (5 files)
  - docs/supported_models/multimodal_language_models.md

**Rationale:** Text-only DeepSeek R1 target requires no image/video generation capabilities. This was a complete parallel subsystem for diffusion models.

### Batch 10: Remaining Platform and Model Documentation
**Removed:** 7 files (~1,053 lines)
- **Platform docs (2 files):**
  - docs/platforms/cpu_server.md (316 lines) - CPU server deployment guide
  - docs/platforms/nvidia_jetson.md (80 lines) - Edge device deployment
- **Model docs (5 files):**
  - docs/basic_usage/glm45.md, glmv.md (208 lines) - GLM models
  - docs/basic_usage/gpt_oss.md (129 lines) - GPT-OSS model
  - docs/basic_usage/minimax_m2.md (66 lines) - Minimax model
  - docs/basic_usage/openai_api_vision.ipynb (254 lines) - Vision API examples

**Rationale:** Text-only target (no vision), NVIDIA GPU multi-node cluster only (no CPU/edge).

---

## 🎯 Current State

### Remaining DeepSeek Files:
✅ **Model Files (5):**
- `python/sglang/srt/models/deepseek.py` - Base model
- `python/sglang/srt/models/deepseek_v2.py` - v2/R1 (PRIMARY)
- `python/sglang/srt/models/deepseek_nextn.py` - Variant
- `python/sglang/srt/models/registry.py` - Model registry
- `python/sglang/srt/models/utils.py` - Utilities

✅ **Shared Components:**
- `python/sglang/srt/models/deepseek_common/` - MLA, MoE infrastructure

✅ **Function Calling:**
- `python/sglang/srt/function_call/` - Tool calling functionality

✅ **Benchmarks:**
- `benchmark/deepseek_v3/` - DeepSeek v3 validation

---

## 🧪 Testing & Validation

### Test Suite Created:
📁 **tests/** directory (outside sglang/)
- ✅ `test_syntax.py` - Python syntax validation
- ✅ `test_no_broken_refs.py` - Broken reference detection
- ✅ `test_imports.py` - Import validation (requires dependencies)
- ✅ `run_all_tests.sh` - Automated test runner

### Validation Results:
✅ **All tests passing:**
- Syntax tests: PASS
- Broken reference tests: PASS
- DeepSeek models import correctly
- No imports of removed models

---

## 📝 Next Steps

### Remaining Cleanup (Phase 2):

1. **AMD GPU (ROCm/HIP) Code** (Deferred to Phase 3)
   - Conditional code in ~72 files
   - Requires careful analysis
   - Will be addressed in deep cleanup phase

2. **Potential Additional Cleanup:**
   - Check for dead imports in remaining files
   - Remove unused test directories
   - Clean up examples/documentation for removed models
   - Verify no broken references in runtime code

### Phase 3 Preparation:
- Build comprehensive dependency graph
- Identify unused layers and kernels
- Map DeepSeek's actual CUDA kernel usage
- Plan conservative removal strategy

---

## 🔍 Key Decisions Made

1. **Text-only DeepSeek R1:** Removed all multimodal variants
2. **NVIDIA GPU only:** Removed NPU/Ascend backends
3. **Conservative approach:** Keep infrastructure, remove only provably unused code
4. **Test-driven:** Validate after each batch
5. **Git checkpoints:** Commit after each batch for easy rollback

---

## 📈 Progress Metrics

### Original Codebase:
- ~663K lines of code
- 1945+ Python files

### After Phase 2 Removals:
- **~205,812 lines removed** (~31% reduction)
- **765 files removed**
- Core DeepSeek functionality preserved

### Estimated Final Target:
- 40-60% code reduction (Phase 2)
- 60-70% total reduction (after Phase 3)

---

## 🚀 Deployment Readiness

### What's Preserved:
✅ All API endpoints and server code
✅ Runtime system (scheduler, memory manager, KV cache)
✅ Model loader and registry
✅ DeepSeek model implementations
✅ NVIDIA CUDA kernels
✅ Distributed inference infrastructure
✅ Testing infrastructure
✅ Launch commands and CLI interface
✅ **Tool calling functionality** (critical requirement)

### Target Deployment:
- Multi-node NVIDIA GPU cluster
- DeepSeek-R1 (text-only)
- Same launch command and API surface

---

## 📋 Documentation Files

- **CLAUDE.md** - Master plan and context
- **REMOVED_FILES.md** - Complete deletion log
- **DEPENDENCIES.md** - Dependency graph
- **PROGRESS_SUMMARY.md** (this file) - Overall progress
- **tests/** - Validation test suite

---

**Status:** Phase 2 ongoing - Ready to continue with additional cleanup or proceed to Phase 3
