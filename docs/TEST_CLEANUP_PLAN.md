# Test Directory Cleanup Plan

**Date:** 2026-01-11 (Night)
**Analysis Agent:** ac04c1a
**Target:** `sglang/python/sglang/test/` (71 files, 19,204 lines)

## Executive Summary

Comprehensive analysis of test directory reveals:
- **Safe to delete:** ~28 files (~5,200 lines) - tests for non-DeepSeek models
- **Must keep:** ~43 files (~14,004 lines) - DeepSeek & generic infrastructure
- **Potential reduction:** 27% of test code

## Current State

- **Total Test Files:** 71 Python files
- **Total Lines:** 19,204
- **Test Categories:**
  - DeepSeek-critical MoE tests (MLA, quantization)
  - Generic infrastructure (attention, quantization, activation)
  - Non-DeepSeek model evaluations (MMLU, MMMU, etc.)
  - Platform-specific code (HIP/AMD)

## Critical Tests to Keep (DeepSeek-Specific)

### 1. MoE Infrastructure (CRITICAL - DeepSeek R1 has 58 MoE layers)
- `test_cutlass_moe.py` (306 lines) - CUTLASS FP8 MoE vs Triton
- `test_cutlass_w16a16_moe.py` (161 lines) - W16A16 MoE quantization
- `test_cutlass_w4a8_moe.py` (285 lines) - W4A8 MoE operations

### 2. MLA Tests (CRITICAL - DeepSeek uses Multi-head Latent Attention)
- `attention/test_flashattn_mla_backend.py` (330 lines) - FlashAttention with MLA
- `attention/test_trtllm_mla_backend.py` - TRT-LLM MLA backend
- `attention/test_flashattn_backend.py` (200+ lines) - Standard attention
- `attention/test_prefix_chunk_info.py` - Prefix handling

### 3. Quantization Tests (CRITICAL - DeepSeek uses FP4/FP8)
- `test_block_fp8.py` (660 lines) - FP8 quantization comprehensive
- `test_block_fp8_deep_gemm_blackwell.py` (251 lines) - Blackwell GPU FP8
- `test_kvfp4_quant_dequant.py` (~220 lines) - KV cache FP4 (NVIDIA FP4 v2)
- `test_custom_ops.py` (148 lines) - FP8 custom operations
- `test_marlin_utils.py` (171 lines) - Marlin quantization

### 4. Core Test Infrastructure
- `test_utils.py` (2,147 lines) - Contains DeepSeek model constants:
  - `DEFAULT_MLA_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"`
  - `DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST = "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"`
  - `DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST = "nvidia/DeepSeek-V3-0324-FP4"`
- `runners.py` (932 lines) - Test runner framework
- `accuracy_test_runner.py` (268 lines) - Accuracy testing
- `performance_test_runner.py` (231 lines) - Performance testing

## Safe to Delete (Non-DeepSeek Models & Features)

### Phase 1: Model-Specific Evaluations (~2,200 lines)

**Vision-Language Models (NOT DeepSeek focus):**
```bash
# VLM evaluation and fixtures
simple_eval_mmmu_vlm.py                    # 458 lines - MMMU VLM evaluation
server_fixtures/mmmu_fixture.py            # VLM fixture setup
kits/mmmu_vlm_kit.py                       # MMMU testing kit
external_models/custom_qwen2_vl.py         # 22 lines - Qwen2 VL (not DeepSeek)
```

**Generic LLM Benchmarks (Not DeepSeek-specific):**
```bash
# General LLM evaluation benchmarks
simple_eval_mmlu.py                        # MMLU benchmark
simple_eval_math.py                        # Math evaluation
simple_eval_humaneval.py                   # HumanEval code generation
simple_eval_mgsm.py                        # 203 lines - Multilingual GSM
simple_eval_aime25.py                      # AIME 2025 Math
simple_eval_gpqa.py                        # Graduate-level QA
```

**Few-Shot & Other Model Tests:**
```bash
# Few-shot utilities
few_shot_gsm8k.py                          # 149 lines - Few-shot GSM8K
few_shot_gsm8k_engine.py                   # 143 lines - GSM8K engine
gpt_oss_common.py                          # 152 lines - OpenAI OSS models (NOT DeepSeek)
test_deepep_utils.py                       # 219 lines - Deep Energy Pattern utils
get_logits_ut.py                           # Generic logits utility
```

**Estimated deletion:** ~2,200 lines (11% reduction)

### Phase 2: Platform-Specific Code Removal (~200-400 lines)

**HIP/AMD GPU code in test files:**
- `test_activation.py` - Remove `is_hip()` checks, `forward_hip()` references
- `test_custom_ops.py` - Remove HIP conditional branches
- Other files with HIP platform checks

**Approach:** Keep files, remove HIP-specific branches only

### Phase 3: Review & Conditional Deletions (~100-200 lines)

**Files needing review:**
- `server_fixtures/default_fixture.py` - Check if tests DeepSeek or generic models
- `server_fixtures/disaggregation_fixture.py` - Check model specificity
- `longbench_v2/validate_longbench_v2_standalone.py` - Review if DeepSeek-compatible

**Decision criteria:** If tests generic non-DeepSeek models: DELETE

## Files to Keep (Generic Infrastructure Used by DeepSeek)

### Speculative Decoding
- `speculative/test_spec_utils.py` (349 lines) - Draft cache allocation
- `server_fixtures/eagle_fixture.py` - EAGLE speculative decoding

### Generic Evaluation Frameworks
- `simple_eval_common.py` (483 lines) - Common evaluation framework
- `simple_eval_longbench_v2.py` (344 lines) - LongBench v2 (works with any model)
- `longbench_v2/test_longbench_v2_eval.py` - LongBench v2 tests
- `run_eval.py` (273 lines) - Evaluation runner

### Constrained Generation (Works with DeepSeek)
- `kits/json_constrained_kit.py` - JSON constraints
- `kits/regex_constrained_kit.py` - Regex constraints
- `kits/matched_stop_kit.py` - Matched stop tokens
- `kits/radix_cache_server_kit.py` - Radix cache

### Other Generic Infrastructure
- `test_programs.py` (578 lines) - SGL program tests
- `test_deterministic.py` (690 lines) - Determinism testing
- `lora_utils.py` (834 lines) - LoRA utilities (fine-tuning)
- `vlm_utils.py` (590 lines) - VLM infrastructure (may be used for multimodal DeepSeek)
- `kl_test_utils.py` (326 lines) - KL divergence testing
- `nightly_utils.py` (336 lines) - Nightly test utilities
- `nightly_bench_utils.py` (174 lines) - Nightly benchmarking

### CI Infrastructure
- `ci/ci_utils.py` - CI test utilities
- `ci/ci_register.py` - CI registry
- `ci/run_with_retry.py` - Retry logic
- `run_combined_tests.py` (200 lines) - Test combination runner

### Activation & Layer Tests
- `test_activation.py` (106 lines) - GeluAndMul, QuickGELU (remove HIP parts)
- `test_layernorm.py` (185 lines) - Layer normalization

### Other Utilities
- `send_one.py` (209 lines) - Single prompt benchmarking
- `test_dump_metric.py` (132 lines) - Metric dumping
- `test_dynamic_grad_mode.py` - Dynamic gradient mode
- `doc_patch.py` - Documentation patching

## Implementation Plan

### Step 1: Safe Deletions (Low Risk)
```bash
cd sglang/python/sglang/test

# Delete VLM-specific tests
rm simple_eval_mmmu_vlm.py
rm server_fixtures/mmmu_fixture.py
rm kits/mmmu_vlm_kit.py
rm external_models/custom_qwen2_vl.py

# Delete generic benchmarks
rm simple_eval_mmlu.py
rm simple_eval_math.py
rm simple_eval_humaneval.py
rm simple_eval_mgsm.py
rm simple_eval_aime25.py
rm simple_eval_gpqa.py

# Delete few-shot & other models
rm few_shot_gsm8k.py
rm few_shot_gsm8k_engine.py
rm gpt_oss_common.py
rm test_deepep_utils.py
rm get_logits_ut.py
```

**Estimated deletion:** 16 files, ~2,200 lines

### Step 2: HIP Code Removal (Files to Edit)
Use automated script or manual editing to remove HIP conditionals:
- `test_activation.py` - Remove `is_hip()`, `forward_hip()` references
- `test_custom_ops.py` - Remove HIP branches
- Search for `is_hip()` across all test files and remove branches

**Estimated line reduction:** ~200-400 lines

### Step 3: Review & Conditional Deletions
Manually review and delete if not DeepSeek-related:
- `server_fixtures/default_fixture.py`
- `server_fixtures/disaggregation_fixture.py`
- `longbench_v2/validate_longbench_v2_standalone.py`

**Estimated deletion:** 1-3 files, ~100-200 lines

## Expected Results

### Before Cleanup
- Files: 71
- Lines: 19,204
- Categories: Mixed (DeepSeek + generic + non-DeepSeek models)

### After Cleanup
- Files: ~52-55 (deletion of 16-19 files)
- Lines: ~16,500-17,000 (reduction of ~2,500-2,700 lines)
- Reduction: 13-14%
- Categories: DeepSeek-specific + generic infrastructure only

### Maintained Test Coverage
✅ **MoE infrastructure** - All tests preserved (critical for DeepSeek R1)
✅ **MLA attention** - All tests preserved (DeepSeek v2/v3/R1 architecture)
✅ **Quantization** - All FP4/FP8 tests preserved (NVIDIA FP4 v2)
✅ **Generic infrastructure** - Attention, activation, layer tests preserved
✅ **Evaluation frameworks** - Keep generic frameworks usable with DeepSeek
✅ **CI/testing utils** - All infrastructure preserved

❌ **Non-DeepSeek models** - Remove all tests specific to other models
❌ **VLM-only tests** - Remove (DeepSeek focus is text models)
❌ **HIP/AMD code** - Remove all platform-specific branches
❌ **Generic benchmarks** - Remove non-DeepSeek-specific evaluations

## Validation Steps

### After Deletions
1. **Verify critical tests remain:**
```bash
# Check MoE tests exist
ls test_cutlass_*moe*.py

# Check MLA tests exist
ls attention/test_*mla*.py

# Check quantization tests exist
ls test_*fp*.py test_block_fp8*.py
```

2. **Verify no broken imports:**
```bash
# Check test discovery
pytest --collect-only sglang/test/
```

3. **Run DeepSeek-specific tests:**
```bash
# Run MoE tests
pytest test_cutlass_moe.py -v

# Run MLA tests
pytest attention/test_flashattn_mla_backend.py -v

# Run quantization tests
pytest test_block_fp8.py -v
```

## Risk Assessment

### Low Risk (Safe to delete immediately)
- VLM-specific tests (not DeepSeek focus)
- Non-DeepSeek model benchmarks
- Few-shot utilities for other models
- GPT-OSS model tests

### Medium Risk (Review before deleting)
- Server fixtures (check if DeepSeek-compatible)
- LongBench standalone validation
- Generic evaluation utilities (may be useful for DeepSeek)

### High Risk (DO NOT DELETE)
- MoE tests (critical for DeepSeek R1)
- MLA tests (critical for DeepSeek architecture)
- Quantization tests (critical for FP4/FP8)
- Core test infrastructure (test_utils.py, runners.py)

## Next Actions

1. **Create automated cleanup script** for Phase 1 deletions
2. **Run script** to delete 16 non-DeepSeek test files
3. **Manual HIP removal** from remaining test files (Phase 2)
4. **Review** server fixtures and decide on Phase 3 deletions
5. **Validate** test suite still runs with `pytest --collect-only`
6. **Commit** changes with detailed documentation

## References

- **Analysis Agent ID:** ac04c1a (for resuming agent if needed)
- **Test Directory:** `sglang/python/sglang/test/`
- **Original Size:** 71 files, 19,204 lines
- **Target Size:** ~52-55 files, ~16,500-17,000 lines
