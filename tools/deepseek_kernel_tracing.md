# DeepSeek Kernel Tracing Tool

When user says "do deepseek kernel tracing", follow these steps to trace all kernel calls back to their **op call sites** in DeepSeek model code.

**Output:** `results/deepseek_kernel_traces.md` - Kernel call chains from kernel call site → op call site (in deepseek_v2.py, etc.)

---

## Key Concept: Kernel vs Op

- **Kernel:** A low-level compute function (e.g., `cutlass_mla_decode`, `silu_and_mul`, `fused_moe_kernel`)
  - Implemented in CUDA (sgl-kernel), FlashInfer, or Triton
  - Call site is in layer code (e.g., `cutlass_mla_backend.py:274`)
  - Represents HOW computation is executed

- **Op (Operation):** A logical operation in the model (e.g., `self.self_attn()`, `self.mlp()`)
  - Call site is in DeepSeek model files (deepseek_v2.py, deepseek.py, etc.)
  - Represents WHAT the model does
  - Multiple kernels may implement the same op (different backends)

---

## Goal

Trace each kernel call up to the top-level model forward call. Op and sub-op names are discovered through tracing.

---

## ⚠️ CROSS-TOOL ALERTS

When running this tool, if you find issues that indicate problems in the **op traces**:

| Issue Found | Action |
|-------------|--------|
| Op call site doesn't exist at stated line | **UPDATE op traces** - fix the line number |
| Op missing from op traces but kernels trace to it | **UPDATE op traces** - add the missing op |
| Op listed in wrong execution order | **UPDATE op traces** - reorder ops |

**Alert format:**
```
⚠️ OP TRACE ISSUE: [description]
Action: Update results/deepseek_op_traces.md - [specific fix needed]
```

**After generating kernel traces:**
- If `results/deepseek_op_traces.md` exists, cross-validate against it
- Report any discrepancies found

---

## Call Chain Format

**IMPORTANT:** The call chain format shows:
1. **First line:** The kernel call site (file:line where the kernel function is invoked)
2. **Following lines:** Trace BACK through callers
3. **Last line:** The **op call site** in DeepSeek model code (deepseek_v2.py, etc.)

### Toy Example

```python
# kernel_lib/ops.py
def some_kernel():          # kernel definition (NOT shown in trace)
    ...

# layers/some_layer.py
def layer_forward():
    ...
    some_kernel()           # line 50: KERNEL CALL SITE (first line of trace)

# layers/wrapper.py
def forward():
    ...
    self.layer.forward()    # line 100: intermediate caller

# models/deepseek_v2.py
def forward():
    ...
    self.some_op()          # line 200: OP CALL SITE (last line of trace)
```

The call chain should be:

```
[some_kernel()](../path/to/layers/some_layer.py#L50) *(sgl-kernel)*
└─> [self.layer.forward()](../path/to/layers/wrapper.py#L100)
    └─> [self.some_op()](../sglang/python/sglang/srt/models/deepseek_v2.py#L200) ← **OP CALL SITE**
```

**Key points:**
- Link format: `[function_name()](path#Lline)` - function name is clickable, links to call site
- First line: kernel call site (where `some_kernel()` is called)
- Last line: op call site (where `self.some_op()` is called in model code)
- Each line shows **where the function was called from** (call site), not where it's defined
- **Indentation shows depth** - each child is indented 4 more spaces than its parent

---

## Step 1: Identify Kernel Sources

Kernels can come from these locations:

### 1.1 sgl-kernel (CUDA kernels)
```bash
# Python bindings - list all modules
ls sglang/sgl-kernel/python/sgl_kernel/*.py

# Find all sgl-kernel imports in sglang (layers AND models)
grep -rn "from sgl_kernel\|import sgl_kernel" --include="*.py" sglang/python/
```

**IMPORTANT:** Run these commands and examine **every result**. Each import reveals an sgl-kernel function. Do not assume you know all kernels - discover them from the grep output.

**⚠️ CRITICAL: Multi-line Import Blocks**

The grep command only shows the line with `from sgl_kernel import`, NOT the actual kernel names in multi-line imports. Use `-A 10` to capture the full import block:

```bash
# Show 10 lines after each import to capture multi-line blocks
grep -rn -A 10 "from sgl_kernel import" --include="*.py" sglang/python/
```

### 1.2 FlashInfer (local repo)
```bash
# Local flashinfer repo at: flashinfer/
# Use -A 10 to capture multi-line import blocks
grep -rn -A 10 "from flashinfer\|import flashinfer" --include="*.py" sglang/python/
```

**IMPORTANT:** Run this grep command and examine **every result**. Each import reveals a flashinfer kernel. Do not assume you know all kernels - discover them from the grep output. Look for imports from different flashinfer submodules (e.g., `flashinfer.decode`, `flashinfer.prefill`, `flashinfer.fused_moe`, `flashinfer.rope`).

**⚠️ Multi-line imports:** Same as sgl-kernel - use `-A 10` or read the file to see the full import block.

### 1.3 Triton JIT Kernels
```bash
# Look for @triton.jit decorators
grep -rn "@triton.jit" --include="*.py" sglang/python/
```

**IMPORTANT:** Run this grep command and examine **every result**. Each `@triton.jit` decorated function is a triton kernel.

---

## Step 2: Trace Kernel Call Sites to Model Code

For each kernel discovered in Step 1, trace from where it's called back to the model code. The call site you end at is the op or sub-op.

### 2.1 Find kernel call site

```bash
# Example: find where cutlass_mla_decode is called
grep -rn "cutlass_mla_decode(" --include="*.py" sglang/python/
# Result: cutlass_mla_backend.py:274
```

### 2.2 Check if kernel is called directly in model code

**FIRST**, check if the kernel is called directly in DeepSeek model files:

```bash
# Check if kernel is called directly in model files
grep -rn "kernel_name(" --include="*.py" sglang/python/sglang/srt/models/deepseek*.py
```

**If found in model code:** The kernel is called directly - no tracing needed. The call site IS the op. Examples:
- `bmm_fp8()` called at deepseek_v2.py:1521, 1632, 1705, 1833
- `dsv3_fused_a_gemm()` called at deepseek_v2.py:1296
- `dsv3_router_gemm()` called at deepseek_v2.py:330
- `merge_state_v2()` called at deepseek_v2.py:1886
- `concat_mla_k()` called at deepseek_v2.py:2046

For these, the call chain is just one level - the kernel call site in the model code.

### 2.3 Trace back through callers (for layer-based kernels)

**If kernel is called in layer code**, trace back to model code:

1. Read the file at the kernel call site - what function contains it?
2. Search for calls to that function
3. Continue until reaching an op call site in DeepSeek model code

```bash
# What function is at cutlass_mla_backend.py:274?
# → CutlassMLABackend.forward_decode()

# What calls forward_decode()?
grep -rn "forward_decode\|\.forward(" sglang/python/sglang/srt/layers/
# → radix_attention.py calls attn_backend.forward()

# What calls RadixAttention.forward()?
grep -rn "self\\.attn\\|self_attn" sglang/python/sglang/srt/models/deepseek*.py
# → deepseek_v2.py:2204 calls self.self_attn() ← MODEL CODE REACHED
```

### 2.3 Trace up to top-level model forward

**Continue tracing until you reach the top-level model forward call** (e.g., `self.model()` in the CausalLM class). The complete call chain reveals:
- **Op/sub-op names** - discovered at each call site in model code
- **Call hierarchy** - how ops relate to each other (e.g., `self.gate()` is called within `self.mlp()`)

DeepSeek model files:
- `deepseek_v2.py` - DeepSeek v2/v3/R1
- `deepseek.py` - DeepSeek v1
- `deepseek_nextn.py` - NextN speculative decoding

---

## Step 3: Build Call Chain with Links

Format each trace with VSCode-clickable links. Use **indentation** (4 spaces per level) to show depth:

```
[kernel_function()](../path/kernel_file.py#Lline) *(source)*
└─> [caller_function()](../path/intermediate.py#Lline)
    └─> [higher_caller()](../path/intermediate.py#Lline)
        └─> [self.op_name()](../sglang/python/sglang/srt/models/deepseek_v2.py#Lline) ← **OP**
```

**For branching paths (multiple siblings):**

```
[kernel_function()](../path/kernel_file.py#Lline) *(source)*
├─> [path_a()](../path/a.py#Lline)
│   └─> [self.op_a()](../sglang/python/sglang/srt/models/deepseek_v2.py#Lline) ← **OP**
└─> [path_b()](../path/b.py#Lline)
    └─> [self.op_b()](../sglang/python/sglang/srt/models/deepseek_v2.py#Lline) ← **OP**
```

**Link format:** `[function_name()](path#Lline)`
- The function name is the clickable text
- Clicking takes you to where that function is called (the call site)

**Tree characters:**
- `└─>` - last/only child at this level
- `├─>` - sibling with more siblings below
- `│` - vertical connector between siblings (continues through children)

**Rules:**
- Use `../` prefix for paths (output file is in `results/` directory)
- Note kernel source in italics on first line: *(sgl-kernel)*, *(flashinfer)*, *(triton)*
- Mark the final op call site clearly with ← **OP**
- **Uniform arrow length** - all arrows are `└─>` or `├─>` (same length)
- **Progressive indentation** - each child indented 4 more spaces than parent
- **NO SHORTCUTS:** Always list full call chains explicitly. Do NOT use "(same as Path X)" or similar.

---

## Step 4: Generate Output File

Create `results/deepseek_kernel_traces.md` with:

1. **Summary Table** - List of all kernels with their source and target op
2. **Detailed Call Chains** - Full call chain for each kernel (kernel call site → op call site)
3. **Kernel Source Breakdown** - Grouped by source (sgl-kernel, flashinfer, triton)

### Output Structure

```markdown
# DeepSeek Kernel Traces

## Summary Table

| Kernel | Source | Target Op | Kernel Call Site |
|--------|--------|-----------|------------------|
| kernel_a | sgl-kernel | discovered_op_1() | file_a.py:50 |
| kernel_b | flashinfer | discovered_op_2() | file_b.py:100 |

## Detailed Call Chains

### 1. [Category discovered from tracing]

#### 1.1 kernel_a *(sgl-kernel)*

[kernel_a()](../path/to/file_a.py#L50) *(sgl-kernel)*
└─> [intermediate()](../path/to/intermediate.py#L75)
    └─> [discovered_op_1()](../sglang/python/sglang/srt/models/deepseek_v2.py#L200) ← **OP**

#### 1.2 kernel_with_branches *(sgl-kernel)*

[kernel_with_branches()](../path/to/file.py#L100) *(sgl-kernel)*
├─> [path_a()](../path/to/a.py#L50)
│   └─> [self.op_a()](../sglang/python/sglang/srt/models/deepseek_v2.py#L300) ← **OP**
└─> [path_b()](../path/to/b.py#L60)
    └─> [self.op_b()](../sglang/python/sglang/srt/models/deepseek_v2.py#L400) ← **OP**
```

---

## Step 5: Verification

After generating the trace file:

```bash
# Verify all referenced files exist
grep -o "../sglang/[^)#]*\|../flashinfer/[^)#]*" results/deepseek_kernel_traces.md | sort -u | while read f; do
  test -f "${f#../}" || echo "MISSING: $f"
done

# Verify all traces end at model code (deepseek*.py)
grep -E "└─> .*deepseek" results/deepseek_kernel_traces.md
```

### 5.1 Verify all discovered kernels are documented

For each kernel discovered in Step 1, verify it appears in the traces file using simple grep:

```bash
# For each kernel name from Step 1, check if it's in the traces file
grep "bmm_fp8" results/deepseek_kernel_traces.md
grep "dsv3_fused_a_gemm" results/deepseek_kernel_traces.md
grep "cutlass_mla_decode" results/deepseek_kernel_traces.md
# ... repeat for each discovered kernel
```

If grep returns nothing, that kernel is missing and needs to be added.

**⚠️ Keep verification simple:**
- ✅ `grep "kernel_name" file.md` - simple, reliable
- ❌ `sed`, `awk`, or complex regex to extract/compare - error-prone

### 5.2 Verify links point to actual call sites

**CRITICAL:** Each link `[function_name()](path#Lline)` must point to a line that **actually contains a call** to that function. Common errors:
- Line is a function definition (`def function_name`)
- Line is empty or unrelated code
- Line number is wrong

```bash
# Extract all links and verify each one
grep -oE "\[([^\]]+)\]\(\.\./[^)]+#L([0-9]+)\)" results/deepseek_kernel_traces.md | while read link; do
  # Extract function name and file path with line number
  func_name=$(echo "$link" | sed 's/\[//;s/\].*//;s/()$//')
  file_path=$(echo "$link" | sed 's/.*(\.\.\///;s/#L.*//')
  line_num=$(echo "$link" | grep -oE "#L[0-9]+" | sed 's/#L//')

  # Get the line content
  if [ -f "$file_path" ]; then
    line_content=$(sed -n "${line_num}p" "$file_path")
    # Check if line contains the function name (as a call, not def)
    if ! echo "$line_content" | grep -q "$func_name"; then
      echo "ERROR - Line $line_num in $file_path does not contain '$func_name'"
      echo "  Link: $link"
      echo "  Line: $line_content"
    elif echo "$line_content" | grep -q "^\s*def "; then
      echo "ERROR - Line $line_num is a definition, not a call site"
      echo "  Link: $link"
      echo "  Line: $line_content"
    fi
  fi
done
```

If errors are found:
1. Find the correct call site: `grep -n "function_name(" file.py | grep -v "def "`
2. Update the link to point to the correct line number

**Example errors:**
```
# Wrong - points to definition:
[self._merge_q()](../path/file.py#L100)  # Line 100: "def _merge_q(self, ..."

# Wrong - points to empty/unrelated line:
[concat_mla_absorb_q()](../path/file.py#L1210)  # Line 1210: ")"

# Correct - points to actual call:
[concat_mla_absorb_q()](../path/file.py#L1215)  # Line 1215: "return concat_mla_absorb_q(q_nope, q_rope)"
```

---

## Repository Locations

- **sglang:** `sglang/` (git submodule)
- **flashinfer:** `flashinfer/` (cloned from https://github.com/lpc0220/flashinfer)
- **sgl-kernel:** `sglang/sgl-kernel/`

---

## Step 6: Cross-Validation with Op Traces

After generating kernel traces, validate against op traces if they exist:

```bash
ls -la results/deepseek_op_traces.md
```

If op traces exist, check that:
1. Every kernel call chain ends at a valid op in the op traces
2. Kernels implementing the same op trace to the same call site

Report discrepancies using the alert format from "CROSS-TOOL ALERTS" section.

---

## Notes

- Focus on **compute-intensive** kernels (attention, MoE, GEMM, normalization, activation)
- Always show the **actual call site** first, not the function definition
- Use **relative paths with ../** for VSCode clickable links
- Note **kernel source** in italics: *(sgl-kernel)*, *(flashinfer)*, *(triton)*
- **Uniform arrows** - use `└─>` and `├─>` (same length), depth shown by indentation
- **Progressive indentation** - each child indented 4 more spaces than parent
- **Vertical connectors** - use `│` to connect sibling branches through their children
- **NO SHORTCUTS:** Always list full call chains explicitly. Each kernel path must show the complete call chain from kernel to op.
- **Direct model kernels:** Some kernels (bmm_fp8, dsv3_fused_a_gemm, dsv3_router_gemm, merge_state_v2, concat_mla_k) are called directly in model code - these have single-level call chains
