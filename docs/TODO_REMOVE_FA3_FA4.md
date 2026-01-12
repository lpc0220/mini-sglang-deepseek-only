# TODO: Remove FA3/FA4 Kernels and Code

**Priority:** High
**Status:** Pending
**Identified:** 2026-01-11

## Objective
Remove FlashAttention 3 (FA3) and FlashAttention 4 (FA4) kernels and related code as they are not top-performant for DeepSeek models.

## Rationale
- FA3/FA4 are not the most performant attention kernels
- Keeping only top-performing kernels reduces compilation time
- Reduces code complexity and binary size
- DeepSeek models may have specific attention optimizations that work better

## Scope

### 1. sgl-kernel (CUDA Kernels)
**Directory:** `sglang/sgl-kernel/csrc/`

Files to investigate:
- `csrc/flashattention/` - Check for FA3/FA4 specific implementations
- Look for files with "fa3", "fa4", "flash_attn_3", "flash_attn_4" in names
- Check CMakeLists.txt for FA3/FA4 kernel compilation

**Action:**
```bash
# Find FA3/FA4 related kernel files
cd sglang/sgl-kernel
find csrc/ -name "*fa3*" -o -name "*fa4*" -o -name "*flash*3*" -o -name "*flash*4*"
grep -r "FA3\|FA4\|flash.*3\|flash.*4" csrc/ CMakeLists.txt
```

### 2. sglang Python Code
**Directory:** `sglang/python/sglang/srt/`

Areas to check:
- `sglang/srt/layers/attention/` - Attention backend implementations
- Look for FA3/FA4 backend classes
- Check for imports from flashinfer or other FA3/FA4 libraries
- Model configuration files that reference FA3/FA4

**Action:**
```bash
cd sglang/python
grep -r "FA3\|FA4\|flash.*attn.*3\|flash.*attn.*4" --include="*.py" sglang/srt/
grep -r "flashinfer.*3\|flashinfer.*4" --include="*.py" sglang/srt/
```

### 3. Dependencies
Check `pyproject.toml` for FA3/FA4 specific dependencies:
- `flashinfer` versions that include FA3/FA4
- Any FA3/FA4-specific packages

### 4. Configuration & Server Args
- `sglang/srt/server_args.py` - Check for FA3/FA4 backend options
- Model configs - Check for FA3/FA4 attention mechanism configs

## Retention Strategy

**Keep:**
- ✅ FlashAttention 2 (FA2) - Current top performer
- ✅ Triton attention implementations
- ✅ Any DeepSeek-specific MLA (Multi-head Latent Attention) implementations
- ✅ Standard attention fallbacks

**Remove:**
- ❌ FlashAttention 3 (FA3) kernels and code
- ❌ FlashAttention 4 (FA4) kernels and code
- ❌ FA3/FA4-specific configuration options
- ❌ FA3/FA4 backend classes

## Implementation Plan

### Phase 1: Discovery (30 minutes)
1. Search for all FA3/FA4 references in codebase
2. Create comprehensive list of files to modify
3. Identify dependencies and potential breaking changes
4. Check if any DeepSeek models explicitly use FA3/FA4

### Phase 2: Kernel Removal (sgl-kernel)
1. Remove FA3/FA4 kernel source files from `csrc/`
2. Update CMakeLists.txt to remove FA3/FA4 compilation
3. Remove FA3/FA4 from Python bindings
4. Test sgl-kernel compilation

### Phase 3: Python Code Removal (sglang)
1. Remove FA3/FA4 attention backend implementations
2. Remove FA3/FA4 imports and references
3. Update server args to remove FA3/FA4 options
4. Remove FA3/FA4 from model configs
5. Update documentation

### Phase 4: Testing
1. Verify sgl-kernel compiles without FA3/FA4
2. Test DeepSeek model loading
3. Run attention layer tests
4. Benchmark to ensure no performance regression

### Phase 5: Documentation
1. Update CLAUDE.md with FA3/FA4 removal
2. Document which attention backends remain
3. Add to cleanup completion report

## Expected Benefits

### Build Time
- **Current:** Compiling all attention kernels (FA2, FA3, FA4)
- **After:** Only FA2 and Triton attention
- **Estimated speedup:** 10-20% faster sgl-kernel compilation

### Code Reduction
- **Estimated:** 5,000-15,000 lines of kernel code
- **Estimated:** 1,000-3,000 lines of Python code
- **Binary size:** Smaller compiled .so files

### Maintenance
- Fewer attention backends to maintain
- Clearer attention strategy (FA2 for performance)
- Less complexity in attention layer selection logic

## Risks & Mitigation

### Risk 1: DeepSeek Models Use FA3/FA4
**Mitigation:** Check DeepSeek model configs first
```bash
grep -r "fa3\|fa4" sglang/python/sglang/srt/models/deepseek*
```
If found, verify if it's critical or just optional

### Risk 2: Breaking Existing Functionality
**Mitigation:**
- Keep FA2 and Triton backends intact
- Test thoroughly before committing
- Maintain rollback capability

### Risk 3: Future Model Compatibility
**Mitigation:**
- Document that FA2 is the primary attention backend
- If future models need FA3/FA4, can be re-added from original SGLang

## Commands to Execute (After Kernel Compilation Completes)

### Step 1: Analyze FA3/FA4 Usage
```bash
# In sgl-kernel
cd sglang/sgl-kernel
find . -name "*fa3*" -o -name "*fa4*" | tee fa34_kernels.txt
grep -r "FA3\|FA4" csrc/ CMakeLists.txt | tee fa34_references.txt

# In Python code
cd ../python
grep -r "fa3\|fa4\|FA3\|FA4" --include="*.py" sglang/ | tee fa34_python.txt
```

### Step 2: Create Removal Script
Based on findings, create `scripts/remove_fa34.py`

### Step 3: Execute Removal
Only after verification that FA3/FA4 are not critical for DeepSeek

## Related Tasks

- [ ] Investigate which attention backend DeepSeek models actually use
- [ ] Check if MLA (Multi-head Latent Attention) uses FA3/FA4
- [ ] Verify FlashAttention 2 performance is sufficient
- [ ] Document final attention backend strategy

## Status Updates

- **2026-01-11:** TODO created during GB200 kernel compilation
- **Next:** Wait for kernel compilation to complete, then analyze FA3/FA4 usage

---

**Note:** Do NOT remove FA3/FA4 until:
1. Current kernel compilation succeeds
2. DeepSeek models confirmed to not use FA3/FA4
3. Thorough analysis of codebase dependencies complete
