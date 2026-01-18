# Dead Code Analysis Tool

When user says "run dead code analysis", follow these steps.

**Reference:** See `tools/registry.md` for the complete kept/removed list.

---

## IMPORTANT: Loop Until Clean

**This analysis MUST be executed in a loop until the codebase is provably clean.**

### Loop Termination Condition
- Execute steps 1-7 as one "round"
- Track whether any code was edited/deleted in each round
- **STOP ONLY when 2 consecutive rounds have ZERO code edits**
- This proves all dead code has been removed (no cascading dependencies remain)
- After loops complete, execute Step 9 (update CLAUDE.md metrics) once

### Why Loop?
Removing dead code can expose NEW dead code:
- Deleting file A may make function B in file C now unused
- Removing import X may make class Y orphaned
- Each removal round may uncover more dead code

### Round Tracking
At the start of each round, announce:
```
=== Dead Code Analysis Round N ===
Previous round edits: [X files modified / deleted]
```

At the end of each round, summarize:
```
=== Round N Complete ===
Files modified: [list]
Files deleted: [list]
Total edits: [count]
Continue to Round N+1: [Yes/No]
```

---

## Purpose

Find NEW dead code that may have been introduced or missed. This is different from deletion completion analysis:
- **Deletion completion**: Verify REMOVED items have zero references
- **Dead code analysis**: Find NEW unreferenced code to potentially remove

---

## Step 1: Check Registry First

Before removing anything, check `tools/registry.md`:
1. If item is in **REMOVED** list → safe to delete references
2. If item is in **KEPT** list → do NOT delete
3. If item is not in either list → investigate before deleting

---

## Step 2: Scan for Potential Dead Code

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang/python
```

### Check for Stale Platform References
```bash
# Platform detection calls (should return 0 or only stub definitions)
grep -rn "is_npu()\|is_hip()\|is_xpu()\|is_hpu()" --include="*.py" sglang/srt/ | grep -v "def is_"
```

### Check for Dead Imports
```bash
# Imports to known-removed modules
grep -rn "from sglang.*import\|import sglang" --include="*.py" sglang/ | \
  grep -E "(mamba|gguf|flash_attn|FlashAttention|grpc|dllm|awq|gptq)" | grep -v __pycache__
```

### Check for Unused Functions
```bash
# Find function definitions, then verify they're called somewhere
# Example: check if a specific function is used
grep -rn "def function_name" --include="*.py" sglang/
grep -rn "function_name(" --include="*.py" sglang/ | grep -v "def function_name"
```

### Check for Orphaned Files
```bash
# List files and check if they're imported anywhere
ls sglang/srt/layers/quantization/*.py | while read f; do
  basename=$(basename $f .py)
  if [ "$basename" != "__init__" ]; then
    count=$(grep -rn "from.*$basename\|import.*$basename" --include="*.py" sglang/ | grep -v __pycache__ | wc -l)
    if [ "$count" -eq 0 ]; then
      echo "Potentially unused: $f"
    fi
  fi
done
```

### Check for Dead Class/Function References
```bash
# Look for references to known-removed items that might have been missed
grep -rn "AWQ\|GPTQ\|Marlin\|CompressedTensors" --include="*.py" sglang/ | grep -v __pycache__ | grep -v "# " | wc -l
```

---

## Step 3: Verify Before Deletion

Before removing any code:

1. **Check registry.md** - Is it in KEPT list?
2. **Check all references** - Is it really unused?
   ```bash
   grep -rn "ClassName\|function_name" --include="*.py" sglang/
   ```
3. **Check imports** - Is the module imported anywhere?
   ```bash
   grep -rn "from sglang.path.to.module" --include="*.py" sglang/
   ```

---

## Step 4: Delete Dead Code

Order of deletion:
1. Delete entire unused files
2. Remove dead imports from remaining files
3. Remove dead conditionals (if branches that never execute)
4. Remove dead enum values
5. Clean up comments referencing removed code

**Track all edits for the round summary.**

---

## Step 5: Update Registry

After removing code:
1. Add removed items to `tools/registry.md` REMOVED section
2. Remove from KEPT section if it was there

---

## Step 6: Verify Syntax

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang/python

# Check modified files
python -m py_compile sglang/srt/path/to/modified_file.py

# Or check all files
find sglang/srt/ -name "*.py" -exec python -m py_compile {} \;
```

---

## Step 7: Verify No Broken References

```bash
# Should return 0 for deleted items
grep -rn "DeletedClassName" --include="*.py" sglang/ | wc -l
```

---

## Step 8: Loop Decision

After completing Steps 1-7:

1. Count total edits in this round (files modified + files deleted)
2. If edits > 0:
   - Increment round counter
   - Go back to Step 1
   - Continue until 2 consecutive rounds have 0 edits
3. If edits == 0:
   - Check if previous round also had 0 edits
   - If yes: **STOP - Analysis Complete**
   - If no: Do one more round to confirm

### Final Report

When terminating, output:
```
=== Dead Code Analysis COMPLETE ===
Total rounds executed: N
Rounds with edits: [list rounds with edit counts]
Final 2 rounds: 0 edits each (confirmed clean)

Summary of all removals:
- Files deleted: [total count]
- Files modified: [total count]
- [List key items removed]
```

---

## Step 9: Update CLAUDE.md Reduction Summary

**IMPORTANT:** After dead code analysis is complete, update the "Reduction Summary" section in `CLAUDE.md` to reflect the latest code metrics.

### Commands to Get Current Metrics

```bash
cd /Users/lpc/workspace/sglang-deepseek-only

# Total lines in sglang/python/
find sglang/python -name "*.py" -exec cat {} \; | wc -l

# Total lines in sgl-kernel C++/CUDA
find sglang/sgl-kernel/src -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.h" 2>/dev/null | xargs cat 2>/dev/null | wc -l

# Total lines in sgl-kernel Python
find sglang/sgl-kernel/python -name "*.py" | xargs cat | wc -l

# Total lines in sgl-model-gateway
find sglang/sgl-model-gateway -name "*.py" -o -name "*.go" 2>/dev/null | xargs cat 2>/dev/null | wc -l

# Breakdown by module (example for srt/)
find sglang/python/sglang/srt -name "*.py" | xargs cat | wc -l
```

### Update These Sections in CLAUDE.md

1. **Line 13:** Update the summary line
   ```
   - **Original:** ~663K lines → **Current:** ~XXXK lines (~XX% reduction)
   ```

2. **Reduction Summary table (line 68-78):** Update all metrics
   - Current Lines
   - Lines Removed
   - Reduction percentage

3. **Current Code Attribution tables (line 80+):** Update line counts for each component

### Example Update Format

After running the commands, update CLAUDE.md with actual numbers:

```markdown
## Reduction Summary

| Metric | Value |
|--------|-------|
| Original Lines | ~663,000 |
| Current Lines | ~XXX,000 |
| Lines Removed | ~XXX,000 |
| Reduction | ~XX% |
| Files Modified | X,XXX+ |
| Models Removed | ~100+ |
| Platform Conditionals Removed | 0 NPU/HIP/XPU references remain |
```
