# TODO: Remove DeepSeek's FlashMLA Implementation

**Priority:** High
**Status:** Pending
**Identified:** 2026-01-11
**Reference:** https://github.com/deepseek-ai/FlashMLA

## Objective
Remove DeepSeek's FlashMLA implementation from sglang/sgl-kernel and use NVIDIA's faster MLA implementation instead.

## Background

### DeepSeek's FlashMLA
- **Repository:** https://github.com/deepseek-ai/FlashMLA
- **Author:** DeepSeek AI team
- **Purpose:** CUDA kernels for Multi-head Latent Attention (MLA)
- **Status:** Superseded by NVIDIA's implementation

### NVIDIA's MLA Implementation
- **Performance:** Faster than DeepSeek's FlashMLA
- **Platform:** Optimized for NVIDIA GPUs (GB200, H100, etc.)
- **Source:** NVIDIA's optimized kernels (likely in DeepSeek-R1-NVFP4-v2 model)
- **Advantage:** Better hardware optimization, maintained by NVIDIA

## Rationale

1. **Performance:** NVIDIA's implementation is faster
2. **Maintenance:** NVIDIA maintains their kernels
3. **Code reduction:** Remove duplicate/slower MLA implementation
4. **Hardware optimization:** NVIDIA knows their hardware best
5. **Binary size:** Smaller compiled libraries

## Scope

### 1. sgl-kernel (CUDA Kernels)
**Location:** `sglang/sgl-kernel/csrc/`

Find and remove:
- FlashMLA kernel files from DeepSeek
- MLA-related CUDA kernels (.cu files)
- MLA Python bindings

**Search commands:**
```bash
cd sglang/sgl-kernel
find csrc/ -name "*mla*" -o -name "*flashmla*"
grep -r "FlashMLA\|deepseek.*mla" csrc/ CMakeLists.txt
```

### 2. Python Code (sglang)
**Location:** `sglang/python/sglang/srt/`

Find and remove:
- DeepSeek FlashMLA backend imports
- FlashMLA attention backend class
- DeepSeek FlashMLA configuration options

**Search commands:**
```bash
cd sglang/python
grep -r "FlashMLA\|deepseek.*flash.*mla" --include="*.py" sglang/srt/
grep -r "from.*flashmla\|import.*flashmla" --include="*.py" sglang/srt/
```

**Areas to check:**
- `sglang/srt/layers/attention/` - FlashMLA backend
- `sglang/srt/models/deepseek_common/` - MLA implementation
- `sglang/srt/server_args.py` - FlashMLA configuration
- `sglang/srt/configs/` - MLA backend selection

### 3. Dependencies
**File:** `sglang/python/pyproject.toml`

Remove:
- Any `flashmla` package dependencies
- DeepSeek FlashMLA repository references
- FlashMLA build requirements

### 4. CMakeLists.txt
Remove FlashMLA compilation targets and source files.

## What to Keep

### ✅ NVIDIA MLA Implementation
- Keep NVIDIA's optimized MLA kernels (if integrated)
- Keep MLA architecture code (the algorithm itself)
- Keep MLA configuration options (just switch backend)

### ✅ MLA Architecture in DeepSeek Models
- `sglang/srt/models/deepseek_common/mla.py` - The MLA layer itself
- MLA forward/backward logic
- MLA configuration (latent dimensions, etc.)

**Important:** Only remove the **kernel implementation** (FlashMLA), not the MLA architecture logic.

## Implementation Strategy

### Phase 1: Locate DeepSeek FlashMLA Code
```bash
# Find all FlashMLA references
cd sglang/sgl-kernel
find . -iname "*flashmla*" -o -iname "*flash*mla*" | tee flashmla_kernels.txt

cd ../python
grep -r "flashmla\|FlashMLA" --include="*.py" sglang/ | tee flashmla_python.txt
```

### Phase 2: Verify NVIDIA MLA is Available
Check if NVIDIA's MLA implementation is already integrated:
```bash
# Check for NVIDIA MLA kernels
grep -r "nvidia.*mla\|nvfp4.*mla" sglang/

# Check if DeepSeek-R1-NVFP4-v2 has bundled MLA kernels
# This model should include NVIDIA's optimized implementation
```

### Phase 3: Create Removal Script
```python
# scripts/remove_deepseek_flashmla.py

import os
import re
from pathlib import Path

# Files to remove (populate after Phase 1)
kernel_files = [
    # e.g., "csrc/flashmla/flashmla_kernel.cu"
]

python_files_to_modify = [
    # e.g., "sglang/srt/layers/attention/flashmla_backend.py" (delete)
    # e.g., "sglang/srt/server_args.py" (remove FlashMLA option)
]

# Remove kernel files
# Modify Python files to remove FlashMLA imports/references
# Update CMakeLists.txt
```

### Phase 4: Test Without DeepSeek FlashMLA
1. Remove DeepSeek FlashMLA code
2. Recompile sgl-kernel
3. Test DeepSeek model loading
4. Verify MLA attention works (using NVIDIA implementation)
5. Benchmark performance

## Expected Benefits

### Build Time Reduction
- **Estimate:** 5-10% faster sgl-kernel compilation
- **Reason:** Fewer MLA kernel variants to compile

### Code Reduction
- **Kernel code:** ~2,000-5,000 lines (.cu files)
- **Python code:** ~500-1,000 lines
- **Binary size:** Smaller .so files

### Performance Improvement
- **Attention speed:** NVIDIA's MLA is faster than DeepSeek's FlashMLA
- **Memory usage:** Potentially better optimized for NVIDIA GPUs
- **Hardware utilization:** Better tensor core usage on Blackwell/Hopper

### Maintenance
- One MLA implementation (NVIDIA's) instead of two
- Maintained by NVIDIA, not community/DeepSeek
- Less code to audit and maintain

## Verification Steps

### Before Removal
```bash
# Benchmark current MLA performance
python3 benchmark/deepseek_v3/bench_one_batch.py --model-path <deepseek-model>
# Note the throughput/latency
```

### After Removal
```bash
# Verify NVIDIA MLA works
python3 -c "from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM; print('✅ MLA loads')"

# Benchmark with NVIDIA MLA
python3 benchmark/deepseek_v3/bench_one_batch.py --model-path <deepseek-model>
# Should be equal or better performance
```

## Risks & Mitigation

### Risk 1: NVIDIA MLA Not Yet Integrated
**Mitigation:**
- Verify NVIDIA implementation exists first
- If not found, integrate NVIDIA's MLA before removing DeepSeek's
- Alternatively, keep DeepSeek FlashMLA as fallback

### Risk 2: Breaking DeepSeek Model Loading
**Mitigation:**
- Test with actual DeepSeek models before committing
- Keep MLA architecture code intact (only remove kernel)
- Maintain rollback capability

### Risk 3: Performance Regression
**Mitigation:**
- Benchmark before and after
- If NVIDIA MLA is slower (unlikely), rollback
- Document performance characteristics

## Related Removals

This removal pairs well with:
- [TODO_REMOVE_FA3_FA4.md](TODO_REMOVE_FA3_FA4.md) - Remove FA3/FA4 kernels

**Combined benefit:**
- Remove FA3, FA4, and DeepSeek FlashMLA
- Keep: FA2 (general), NVIDIA MLA (DeepSeek-specific), Triton (fallback)
- **Total build time reduction:** 15-30%
- **Total code reduction:** 10,000-20,000 lines

## Investigation Commands (Run After Current Compilation)

```bash
# 1. Find DeepSeek FlashMLA in kernels
cd sglang/sgl-kernel
find csrc/ -iname "*flashmla*" -o -iname "*mla*" | grep -v test
grep -r "deepseek.*flash.*mla" csrc/ CMakeLists.txt

# 2. Find FlashMLA in Python code
cd ../python
grep -r "flashmla\|FlashMLA" --include="*.py" sglang/srt/layers/attention/
grep -r "flashmla\|FlashMLA" --include="*.py" sglang/srt/models/deepseek_common/

# 3. Check for NVIDIA MLA implementation
grep -r "nvidia.*mla\|nvfp4.*mla\|nv.*mla" --include="*.py" --include="*.cu" sglang/

# 4. Check dependencies
grep -i "flashmla" sglang/python/pyproject.toml
```

## Documentation Updates

After removal:
1. Update CLAUDE.md - Note that NVIDIA MLA is used
2. Update docs/MAC_VALIDATION_REPORT.md - Add FlashMLA removal
3. Create docs/MLA_BACKEND_STRATEGY.md - Document MLA implementation choice
4. Update build documentation

## Status Updates

- **2026-01-11:** TODO created based on user's finding that NVIDIA has faster MLA
- **Next:** Wait for current kernel compilation, then investigate FlashMLA usage
- **Then:** Verify NVIDIA MLA is available and working
- **Finally:** Remove DeepSeek FlashMLA if safe

---

**Key Insight:** DeepSeek FlashMLA (https://github.com/deepseek-ai/FlashMLA) is slower than NVIDIA's implementation. Removing it will:
1. Speed up compilation
2. Improve runtime performance
3. Reduce code complexity
4. Use hardware vendor's optimized kernels

**Priority:** High - significant performance and maintainability win
