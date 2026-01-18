# Module Removal Tool

When user says "follow tools/remove_module.md to delete XXX", follow these steps.

**Reference:** See `tools/registry.md` for the authoritative kept/removed list.

---

## Step 1: Pre-Removal Check

Before removing anything:
1. Check `tools/registry.md` - Is it in KEPT list? If yes, STOP and ask user.
2. Confirm with user if the module should be removed entirely.

---

## Step 2: Find All References

Search for ALL references to the module across the codebase:

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang

# Search in Python files (case-insensitive)
grep -rni "module_name" --include="*.py" python/ sgl-kernel/ | grep -v __pycache__

# Also search for variations (e.g., AWQ vs awq, function names, class names)
```

Create a todo list with all locations that need cleanup.

---

## Step 3: Identify Removal Targets

Categorize findings into:

### 3.1 Files to DELETE entirely
- Dedicated module files (e.g., `awq.py`, `awq_triton.py`)
- Test files for the module (e.g., `test_awq_*.py`)
- Benchmark files for the module (e.g., `bench_awq_*.py`)

### 3.2 Code to REMOVE from files
- Import statements
- Function definitions
- Class definitions
- Conditional branches using the module
- Optional parameters/fields specific to the module

### 3.3 Comments to CLEAN UP
- Docstrings mentioning the module
- Inline comments referencing the module

---

## Step 4: Removal Order

Follow this order to avoid broken imports:

1. **Delete dedicated files** (tests, benchmarks, main module files)
2. **Remove from sgl-kernel** (if applicable):
   - Remove functions from source files (e.g., `gemm.py`, `marlin.py`)
   - Remove exports from `__init__.py`
3. **Remove from main codebase**:
   - Remove imports
   - Remove function/class definitions
   - Remove usage sites
   - Remove optional fields/parameters
4. **Clean up comments** (docstrings, inline comments)

---

## Step 5: sgl-kernel Cleanup (if applicable)

If the module has sgl-kernel functions:

### 5.1 Find sgl-kernel references
```bash
grep -rn "module_name" --include="*.py" sgl-kernel/python/sgl_kernel/
```

### 5.2 Remove from source files
- `sgl-kernel/python/sgl_kernel/gemm.py` - GEMM operations
- `sgl-kernel/python/sgl_kernel/marlin.py` - Marlin repacking
- Other relevant files

### 5.3 Update __init__.py
Remove exports from `sgl-kernel/python/sgl_kernel/__init__.py`

### 5.4 Delete test/benchmark files
```bash
rm -v sgl-kernel/tests/test_module_*.py
rm -v sgl-kernel/benchmark/bench_module_*.py
```

---

## Step 6: Verify Syntax

Check all modified files compile:

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang

# Check specific files
python -m py_compile path/to/modified/file.py

# Or check all Python files
find python/ sgl-kernel/ -name "*.py" -exec python -m py_compile {} \; 2>&1 | head -20
```

---

## Step 7: Verify Complete Removal

Confirm zero references remain:

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang

# Should return nothing
grep -rni "module_name" --include="*.py" python/ sgl-kernel/ | grep -v __pycache__
```

---

## Step 8: Update Registry

Add the removed module to `tools/registry.md` under the REMOVED section:

```markdown
### Category (e.g., Quantization)
- **ModuleName** - file1.py, file2.py, function_name (description of what was removed)
```

---

## Example: AWQ Removal

### Files Deleted
- `python/sglang/srt/layers/quantization/awq.py`
- `python/sglang/srt/layers/quantization/awq_triton.py`
- `sgl-kernel/tests/test_awq_dequant.py`
- `sgl-kernel/tests/test_marlin_repack.py`
- `sgl-kernel/tests/test_marlin_gemm.py`
- `sgl-kernel/benchmark/bench_awq_dequant.py`

### Code Removed
| File | What was removed |
|------|------------------|
| `deepseek_v2.py` | `awq_dequantize` import, AWQ-compatible code path |
| `marlin_utils.py` | `awq_to_marlin_zero_points()`, `moe_awq_to_marlin_zero_points()`, `apply_awq_marlin_linear()` |
| `fused_moe_triton_kernels.py` | Renamed `fused_moe_kernel_gptq_awq` to `fused_moe_kernel_gptq` |
| `sgl-kernel/gemm.py` | `awq_dequantize()` function |
| `sgl-kernel/marlin.py` | `awq_marlin_repack()`, `awq_marlin_moe_repack()` |
| `sgl-kernel/__init__.py` | `awq_dequantize`, `awq_marlin_repack`, `awq_marlin_moe_repack` exports |
| `moe_runner/marlin.py` | `w13_qzeros`, `w2_qzeros` optional fields |

### Comments Cleaned
| File | Comment removed |
|------|-----------------|
| `moe_wna16.py` | "to align with awq" |
| `linear.py` | "needed to load deepseek v3 awq" |
| `activation.py` | "used for some quantization methods like AWQ" |
| `auto_round.py` | "only used by AWQ" |
| `gptq.py` | "only used by AWQ" (2 occurrences) |
| `base_config.py` | "only used by AWQ" |

### Registry Update
```markdown
- **AWQ** - awq.py, awq_triton.py, awq_dequantize, awq_marlin_repack, awq_marlin_moe_repack (entire AWQ quantization removed including sgl-kernel functions)
```

---

## Checklist

- [ ] Pre-removal check completed (not in KEPT list)
- [ ] All references found with grep
- [ ] Dedicated files deleted
- [ ] sgl-kernel functions removed (if applicable)
- [ ] sgl-kernel __init__.py updated (if applicable)
- [ ] Main codebase cleaned
- [ ] Comments cleaned
- [ ] All modified files pass syntax check
- [ ] Zero references remain (verified with grep)
- [ ] `tools/registry.md` updated
