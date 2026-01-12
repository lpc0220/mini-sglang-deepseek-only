# SGLang DeepSeek-Only Project

## Project Goal
Shrink the SGLang repository (663K+ lines of code across 1945+ Python files) to a minimal codebase that:
- **ONLY** supports DeepSeek models (v2, v3, R1)
- Maintains **100% correctness** - no functionality changes
- Keeps **ALL optimizations** enabled for DeepSeek models
- Targets **NVIDIA GPU only** for multi-node deployment (remove CPU, NPU, AMD GPU support)
- Is testable on Mac (without full model runs) using single-layer testing strategy
- **Preserves the same launch command and API** - server, API endpoints, and runtime infrastructure remain unchanged
- **Iterative removal approach** - only remove unused code, unreferenced code, and non-DeepSeek models

## Codebase Overview
- **Total Lines:** ~663K lines (Python, C++, CUDA)
- **Python Files:** 1945 files
- **Repository Location:** `sglang/` (all SGLang code is in this subdirectory)
- **Main Source:** `sglang/python/sglang/`
- **Key Directories:**
  - `sglang/python/sglang/srt/` - Runtime system
  - `sglang/python/sglang/srt/models/` - Model implementations
  - `sglang/python/sglang/srt/hardware_backend/` - Hardware backends (NPU found)
  - `sglang/python/sglang/jit_kernel/` - JIT compiled kernels
  - `sglang/benchmark/deepseek_v3/` - DeepSeek benchmarks exist!

## Target DeepSeek Model Architecture

### Primary Model: DeepSeek-R1 (NVIDIA FP4 v2)
- **Reference:** https://huggingface.co/nvidia/DeepSeek-R1-NVFP4-v2
- **Total Layers:** 61 layers
  - **58 MoE (Mixture of Experts) layers** - Requires MoE infrastructure
  - **3 Standard transformer layers** - Standard attention/FFN
- **Key Features:**
  - Multi-head Latent Attention (MLA)
  - Mixture of Experts (MoE) routing
  - NVIDIA FP4 quantization support
- **Deployment Target:** Multi-node NVIDIA GPU cluster
- **Testing Strategy:** Single-layer testing on Mac (see Testing Strategy section)

### DeepSeek Model Files Identified
- `sglang/python/sglang/srt/models/deepseek.py`
- `sglang/python/sglang/srt/models/deepseek_v2.py` - DeepSeek v2 architecture
- `sglang/python/sglang/srt/models/deepseek_common/` - Shared components (MLA, MoE)
- `sglang/python/sglang/srt/models/deepseek_nextn.py`
- `sglang/python/sglang/srt/models/deepseek_janus_pro.py` - Multimodal variant
- `sglang/python/sglang/srt/models/deepseek_ocr.py` - OCR variant
- `sglang/python/sglang/srt/models/deepseek_vl2.py` - Vision-Language variant

## Removal Targets (Confirmed)
1. **Hardware Backends:**
   - NPU support (`sglang/python/sglang/srt/hardware_backend/npu/`)
   - CPU backends
   - AMD GPU backends
   - Ascend NPU tests (`sglang/python/sglang/test/ascend/`)

2. **Other Models:** All non-DeepSeek model files in `sglang/python/sglang/srt/models/`

3. **Benchmarks:** All non-DeepSeek benchmarks (keep `sglang/benchmark/deepseek_v3/` only)

## Iterative Shrinking Strategy

**Core Principle:** This is an **iterative, dependency-driven** approach. We only remove code that is:
1. **Provably unused** by DeepSeek models
2. **Unreferenced** in the dependency graph
3. **Non-essential** for the API/server infrastructure

**What We Keep:**
- All API endpoints and server code
- Runtime system (scheduler, memory manager, KV cache)
- Model loader and registry (simplified to DeepSeek only)
- All DeepSeek model implementations and dependencies
- NVIDIA CUDA kernels used by DeepSeek
- Distributed inference infrastructure (multi-node support)
- Testing infrastructure
- Launch commands and CLI interface

**What We Remove:**
- Non-DeepSeek model implementations
- Hardware backends: NPU, CPU-only, AMD GPU
- Unused CUDA kernels (not referenced by DeepSeek)
- Unused layers/attention mechanisms
- Non-DeepSeek benchmarks and examples

### Phase 1: Discovery & Dependency Analysis (Week 1)
**Goal:** Understand what DeepSeek models actually use

1. **Map DeepSeek Dependencies**
   - Trace imports from DeepSeek model files (all 7 variants)
   - Identify required layers, kernels, utilities
   - Document optimizer usage
   - Find DeepSeek-specific features (MLA, MoE, etc.)
   - Map server/API dependencies

2. **Identify Hardware Dependencies**
   - Find CUDA kernel usage in DeepSeek models
   - Map NVIDIA-specific optimizations
   - Locate CPU/NPU/AMD conditional code paths
   - Document inference engine requirements
   - Identify multi-node communication dependencies (NCCL, tensor parallel, etc.)

3. **Test Infrastructure Analysis**
   - Find existing DeepSeek tests
   - Identify minimal test cases (unit tests, not full model)
   - Document current test coverage
   - Plan Mac-compatible testing strategy

4. **Build Reference Graph**
   - Create "keep list" of all DeepSeek dependencies
   - Create "remove list" of provably unused code
   - Identify "uncertain" code requiring further analysis

**Deliverables:**
- Dependency graph (DeepSeek → Layers → Kernels → Hardware)
- Keep list (files/directories to preserve)
- Remove list (files/directories safe to delete)
- Test strategy document

### Phase 2: Safe Removals (Week 2)
**Goal:** Remove obviously unused code with zero risk

**Approach:** Use the "remove list" from Phase 1. After each batch of removals:
1. Run import checks
2. Run existing tests
3. Verify server still launches
4. Document what was removed

**Removal Batches:**

1. **Remove Other Models** (Low Risk - Batch 1)
   - Delete all non-DeepSeek model files in `sglang/python/sglang/srt/models/`
   - Update model registry to only reference DeepSeek models
   - Verify imports still work
   - Test: Import DeepSeek models successfully

2. **Remove Hardware Backends** (Medium Risk - Batch 2)
   - Delete NPU directory: `sglang/python/sglang/srt/hardware_backend/npu/`
   - Remove AMD GPU conditional code
   - Remove CPU-only paths (if any)
   - Keep NVIDIA CUDA code only
   - Test: Check hardware detection still works for NVIDIA GPUs

3. **Remove Non-DeepSeek Benchmarks** (Low Risk - Batch 3)
   - Keep `sglang/benchmark/deepseek_v3/` only
   - Delete all other benchmark directories
   - Test: Verify DeepSeek benchmark still runs

4. **Remove Ascend NPU Tests** (Low Risk - Batch 4)
   - Delete `sglang/python/sglang/test/ascend/`
   - Test: Run remaining test suite

**Validation After Each Batch:**
- Check import tree integrity (`python -m sglang.srt.models.deepseek_v2`)
- Run DeepSeek-specific tests (if any exist)
- Verify no broken references (grep for deleted module names)
- Test server launch command (dry run with --help)

**Deliverables:**
- ~40-60% code reduction
- Passing test suite
- Updated documentation
- REMOVED_FILES.md tracking log

### Phase 3: Deep Cleanup (Week 3)
**Goal:** Remove unused infrastructure and dependencies based on dependency analysis

**Important:** Only remove code that is **provably unreferenced** by DeepSeek models. When in doubt, keep it.

**Approach:** Use dependency graph from Phase 1 to identify unused code.

1. **Layer Cleanup** (Conservative approach)
   - Identify layers NOT imported by any DeepSeek model
   - Remove unused attention mechanisms (non-MLA)
   - Remove model-specific layers (e.g., Llama-specific, GPT-specific)
   - Keep: Core layers (embedding, linear, normalization, MLA, MoE)
   - Test: Import and instantiate DeepSeek model

2. **Kernel Cleanup** (High risk - careful!)
   - Identify CUDA kernels NOT called by DeepSeek forward pass
   - Remove CPU-only kernels (we're NVIDIA GPU only)
   - Remove unused CUDA kernels (verify with call graph)
   - Keep: Flash Attention, MoE kernels, quantization kernels
   - Test: Run single-layer forward pass

3. **Manager/Scheduler Cleanup** (Very conservative)
   - Remove NPU/AMD scheduler code paths
   - Simplify model loader to only support DeepSeek architectures
   - Keep: All memory managers (used by runtime)
   - Keep: Batch scheduler, KV cache manager
   - Test: Server launch and basic inference

4. **Utility Cleanup**
   - Remove model-specific utilities (tokenizers for other models)
   - Keep: Common utilities (logging, config, metrics)
   - Test: Import checks

5. **Multimodal Cleanup** (Conditional)
   - Analyze: Does DeepSeek R1 (text-only) need multimodal infrastructure?
   - Keep: If deepseek_vl2.py or deepseek_janus_pro.py are needed
   - Remove: If only text models are required
   - Decision: Clarify with user if multimodal variants are needed

**Deliverables:**
- ~60-70% code reduction (conservative estimate)
- Simplified architecture
- Updated dependency tree
- All tests passing
- Server launches successfully

### Phase 4: Testing & Validation (Week 4)
**Goal:** Ensure correctness without full model runs

1. **Unit Testing Strategy (Mac-compatible)**
   - Test model architecture construction (no weights)
   - Test tokenizer initialization
   - Test layer forward passes with dummy tensors
   - Test scheduler logic with mock inputs

2. **Integration Testing Strategy**
   - Test API endpoints with small configs
   - Test batching logic
   - Test KV cache operations
   - Mock GPU operations if needed

3. **Documentation**
   - Update README
   - Document removed features
   - Add architecture guide
   - Create deployment guide

**Deliverables:**
- Comprehensive test suite
- All tests passing on Mac
- Complete documentation

## Context Management Strategy

### When Working on Specific Tasks:
1. **Always check this file first** for current phase and status
2. **Update progress** in the relevant phase section
3. **Document decisions** in the "Decisions Log" section below
4. **Track removed files** in the "Removed Files Log"

### File Organization:
- **CLAUDE.md** (this file) - Master plan and context
- **DECISIONS.md** - Detailed technical decisions
- **REMOVED_FILES.md** - Complete log of removed code
- **DEPENDENCIES.md** - DeepSeek dependency graph
- **TESTING.md** - Test strategy and results

## Current Status
- **Phase:** Phase 1 - Round 1 (Initial Survey) - IN PROGRESS
- **Last Updated:** 2026-01-11
- **Lines of Code:** 663,394
- **Next Action:** Begin removing non-DeepSeek models

### Round 1 Progress (2026-01-11)
✅ Created dependency tracking structure (DEPENDENCIES.md, deps/)
✅ Surveyed models directory - identified ~100+ non-DeepSeek models
✅ Created initial keep_list.txt and remove_list.txt
✅ **CRITICAL DISCOVERY:** Tool calling functionality located at `sglang/python/sglang/srt/function_call/`
   - DeepSeek-specific detectors: deepseekv31_detector.py, deepseekv32_detector.py, deepseekv3_detector.py
   - Base infrastructure: base_format_detector.py, core_types.py, function_call_parser.py, utils.py
   - **ACTION:** Keep entire function_call/ directory

## Decisions Log

### 2026-01-11: Initial Setup & Round 1
- Cloned SGLang repository
- Identified 7 DeepSeek model variants (4 text-only, 3 multimodal)
- Confirmed NPU backend exists (removal target)
- Found existing `benchmark/deepseek_v3/` directory
- Located tool calling functionality at `sglang/python/sglang/srt/function_call/`

### 2026-01-11: User Decision - Text-Only R1
**Decision:** Keep ONLY text-only DeepSeek R1 models
- **KEEP:** deepseek.py, deepseek_v2.py, deepseek_nextn.py, deepseek_common/
- **REMOVE:** deepseek_janus_pro.py (multimodal), deepseek_ocr.py (OCR), deepseek_vl2.py (Vision-Language)
- **Rationale:** Focus on DeepSeek-R1 text-only model for multi-node NVIDIA GPU deployment

## Testing Strategy (Mac Environment)

### Challenges:
- Mac doesn't have NVIDIA GPU (deployment target is multi-node NVIDIA cluster)
- Can't run full DeepSeek R1 model (61 layers, 58 MoE layers - too large)
- Need to verify correctness without full model execution
- Must validate MoE routing logic and MLA attention mechanism

### Single-Layer Testing Strategy (Primary Approach):

#### 1. Single Standard Layer Test
- **Goal:** Test 1 of the 3 standard transformer layers
- **Components to validate:**
  - Multi-head Latent Attention (MLA) forward pass
  - FFN (Feed-Forward Network) computation
  - Layer normalization
  - Residual connections
- **Test approach:**
  - Create minimal config with `num_layers=1`
  - Use random/dummy input tensors (e.g., shape `[batch=1, seq_len=32, hidden_dim]`)
  - Verify output shapes and no crashes
  - Compare with original SGLang (same input → same output)

#### 2. Single MoE Layer Test
- **Goal:** Test 1 of the 58 MoE layers
- **Components to validate:**
  - MLA attention mechanism
  - MoE routing logic (expert selection)
  - Expert computation (parallel FFNs)
  - Expert output combining
  - Load balancing mechanisms
- **Test approach:**
  - Create minimal config with `num_layers=1`, `num_experts=8` (reduced from full)
  - Mock expert weights or use small random weights
  - Verify routing decisions are deterministic
  - Check expert load distribution
  - Validate shape transformations

#### 3. Layer Combination Test
- **Goal:** Test 1 standard layer + 1 MoE layer
- **Validates:** Inter-layer communication and KV cache passing
- **Config:** `num_layers=2` (1 standard + 1 MoE)

#### 4. Additional Unit Tests
1. **Architecture Tests:** Verify model construction without weights
2. **Tokenizer Tests:** Test DeepSeek tokenizer initialization
3. **Config Tests:** Validate DeepSeek config parsing (61 layers, MoE params)
4. **Shape Tests:** Verify tensor shapes through the pipeline
5. **Import Tests:** Ensure no broken dependencies
6. **KV Cache Tests:** Test cache initialization and updates (single layer)
7. **Quantization Tests:** Verify FP4 quantization utilities (if used in code)

#### 5. Mac-Specific Adaptations
- Use CPU tensors (PyTorch MPS backend if needed)
- Mock CUDA-specific operations where necessary
- Test with `device='cpu'` override
- Skip actual GPU memory allocation tests
- Focus on **logic correctness**, not performance

### Multi-Node Testing Considerations
Since deployment target is multi-node NVIDIA GPU:
- **Preserve:** Distributed training/inference code
- **Test on Mac:** Single-process logic only
- **Validate on GPU:** Use cloud instance for final multi-node testing
- **Keep:** Tensor parallelism, pipeline parallelism, expert parallelism code

### Future Validation (NVIDIA GPU Cloud):
1. Run full 61-layer model on multi-node GPU cluster
2. Compare outputs with original SGLang (end-to-end)
3. Benchmark performance (throughput, latency)
4. Validate multi-node communication (NCCL, etc.)
5. Test with actual DeepSeek-R1 weights

### Test Success Criteria:
✅ Single layer produces correct output shapes
✅ MoE routing is deterministic and balanced
✅ MLA attention mechanism computes correctly
✅ No import errors or missing dependencies
✅ Config parsing works for DeepSeek models
✅ Code runs on CPU (Mac) without CUDA errors
✅ Final validation: Multi-node GPU run matches original SGLang output

## Key Files to Track

### Always Preserve:
- DeepSeek model files (7 files identified)
- CUDA kernels used by DeepSeek
- Core runtime system
- NVIDIA GPU support
- Test infrastructure

### Safe to Remove:
- NPU backend
- AMD GPU backend
- CPU-only backends
- Non-DeepSeek models (~50+ model files)
- Non-DeepSeek benchmarks
- Unused multimodal code (TBD)

## Questions to Answer in Phase 1
1. Does DeepSeek R1 use multimodal features? (check deepseek_vl2.py, deepseek_janus_pro.py) - **R1 is text-only, multimodal variants separate**
2. What CUDA kernels are actually used by DeepSeek?
3. Are there CPU fallback paths we need to keep? - **No, deployment is NVIDIA GPU only**
4. What dependencies does the MLA (Multi-head Latent Attention) need?
5. What dependencies does the MoE (Mixture of Experts) need? - **Critical: 58/61 layers are MoE**
6. Do we need distributed training code or just inference? - **YES: Multi-node deployment required**
7. What tokenizer does DeepSeek use?
8. How are the 3 standard layers vs 58 MoE layers configured in the model definition?
9. What FP4 quantization infrastructure is needed for NVIDIA FP4 v2?
10. What multi-node communication libraries are used? (NCCL, tensor parallel, pipeline parallel)
11. How to configure single-layer testing without breaking the model architecture?

## Iteration Plan & Safety Principles

**Safety-First Approach:**
1. Work in **small batches** (one directory at a time)
2. **Test after each removal** (no big bang approach)
3. **Document every decision** in this file
4. **Maintain git history** for easy rollback
5. **Create checkpoints** after each phase
6. **Verify before delete:** Always check if code is referenced before removal
7. **Keep when uncertain:** If we're not sure if code is used, keep it and investigate further
8. **Preserve API surface:** Never remove public APIs or server endpoints
9. **Validate incrementally:** After each removal batch, verify:
   - Imports work
   - Tests pass
   - Server launches
   - No broken references

**Tools for Verification:**
- `grep -r "import.*<deleted_module>"` - Check for import references
- `python -c "import sglang.srt.models.deepseek_v2"` - Test imports
- `python -m sglang.launch_server --help` - Verify server launches
- `pytest sglang/python/sglang/test/` - Run test suite
- Git branches for each phase - easy rollback if needed

## Notes
- Original repo: https://github.com/sgl-project/sglang
- Workspace: `/Users/lpc/workspace/sglang-deepseek-only`
- Do not reference: `/Users/lpc/workspace/codefactor/design/TECHNICAL_DESIGN.md` (different project)
