# Single Node GPU Testing Guide

**Date:** 2026-01-11
**Objective:** Compile sgl-kernel on single GPU node and validate CUDA-only build

## Pre-requisites

### Hardware
- NVIDIA GPU (Compute Capability 8.0+, ideally 9.0+ for FP4 support)
- Sufficient VRAM (depends on test size, 24GB+ recommended)

### Software
- CUDA 12.1+ (check: `nvcc --version`)
- Python 3.8+
- GCC/G++ compiler
- CMake 3.18+

## Phase 1: Clone and Setup

```bash
# Clone repository with submodule
git clone --recursive https://github.com/lpc0220/mini-sglang-deepseek-only.git
cd mini-sglang-deepseek-only

# Verify submodule is correct branch
cd sglang
git branch -v
# Should show: * deepseek-only 2cc9ccb13

cd ..
```

## Phase 2: Compile sgl-kernel (CUDA Kernels)

The sgl-kernel contains all CUDA kernels. This is the critical test to verify our CUDA-only build works.

```bash
cd sglang/sgl-kernel

# Check build configuration
cat pyproject.toml | grep -A5 "\[build-system\]"

# Expected: Should only have CUDA dependencies, no CPU/ROCm/XPU configs
# The following files should NOT exist:
# - pyproject_cpu.toml (DELETED)
# - pyproject_rocm.toml (DELETED)
# - setup_rocm.py (DELETED)

# Verify CMakeLists.txt has no deleted kernel references
grep "timestep_embedding\|apply_token_bitmask_inplace\|causal_conv1d\|gguf_kernel" CMakeLists.txt
# Should return nothing (these 4 kernels were removed)

# Install sgl-kernel with CUDA support
pip install -e .
```

### Expected Output (Success)
```
Building wheels for collected packages: sgl-kernel
  Building editable for sgl-kernel (pyproject.toml) ... done
Successfully installed sgl-kernel-0.x.x
```

### Expected Output (Failure - if platform code remains)
```
ERROR: Could not find platform-specific configuration
ERROR: is_npu/is_xpu/is_hip not defined
ERROR: Missing NPU/XPU dependencies
```

### Compilation Warnings to Ignore
- CUDA architecture warnings (if GPU is older)
- Deprecation warnings from PyTorch/CUDA
- Template instantiation warnings

### Compilation Errors to Fix
- ❌ Missing CUDA toolkit
- ❌ Incompatible CUDA version
- ❌ Missing CMake
- ❌ Compiler errors in kernel code

## Phase 3: Verify sgl-kernel Installation

```bash
# Test import
python3 -c "import sgl_kernel; print('✅ sgl-kernel imported successfully')"

# Check available kernels
python3 << 'EOF'
import sgl_kernel
import inspect

# List all available functions
kernels = [name for name, obj in inspect.getmembers(sgl_kernel)
           if not name.startswith('_')]
print(f"✅ Found {len(kernels)} kernels in sgl-kernel")
print("\nAvailable kernels:")
for k in sorted(kernels)[:20]:  # Show first 20
    print(f"  - {k}")
EOF
```

### Expected Kernels (Partial List)
Should include:
- `rmsnorm` (RMS normalization for DeepSeek)
- `fused_add_rmsnorm` (Fused operations)
- `silu_and_mul` (Activation functions)
- `gelu_and_mul`, `gelu_tanh_and_mul`
- `moe_fused_gate` (MoE routing)
- `topk_softmax` (MoE expert selection)
- FP8/quantization kernels

Should NOT include (deleted):
- ❌ `timestep_embedding` (diffusion models - deleted)
- ❌ `apply_token_bitmask_inplace_cuda` (grammar - deleted)
- ❌ `causal_conv1d` (Mamba - deleted)
- ❌ `gguf_kernel` (GGUF format - deleted)

## Phase 4: Compile sglang Python Package

```bash
cd ../python

# Check for platform-specific configs
ls -la pyproject*.toml
# Should only see: pyproject.toml (main CUDA config)
# Should NOT see:
# - pyproject_cpu.toml (DELETED)
# - pyproject_xpu.toml (DELETED)
# - pyproject_other.toml (DELETED)

# Install sglang with CUDA support
pip install -e ".[srt]"
```

### Expected Dependencies (from pyproject.toml)
- torch (with CUDA)
- transformers
- flashinfer (CUDA attention kernels)
- vllm (optional, for some quantization)
- pybase64
- requests, fastapi, uvicorn (server)
- Other standard Python packages

### Should NOT Install (platform-specific)
- ❌ intel-openmp (CPU-only)
- ❌ ROCm/HIP packages
- ❌ NPU/Ascend packages
- ❌ XPU packages

## Phase 5: Test Imports (Critical Validation)

```bash
# Run comprehensive import test
python3 << 'EOF'
import sys

print("Testing sglang imports...")
errors = []

# Core imports
try:
    import sglang
    print("✅ import sglang")
except Exception as e:
    errors.append(f"❌ import sglang: {e}")

try:
    import sglang.srt
    print("✅ import sglang.srt")
except Exception as e:
    errors.append(f"❌ import sglang.srt: {e}")

# Layer imports (where platform code was removed)
try:
    from sglang.srt.layers.activation import SiluAndMul, GeluAndMul
    print("✅ import activation layers")
except Exception as e:
    errors.append(f"❌ import activation: {e}")

try:
    from sglang.srt.layers.layernorm import RMSNorm
    print("✅ import layernorm")
except Exception as e:
    errors.append(f"❌ import layernorm: {e}")

# MoE imports (critical for DeepSeek)
try:
    from sglang.srt.layers.moe import FusedMoE
    print("✅ import MoE layers")
except Exception as e:
    errors.append(f"❌ import MoE: {e}")

try:
    from sglang.srt.layers.moe.topk import TopK
    print("✅ import MoE TopK")
except Exception as e:
    errors.append(f"❌ import TopK: {e}")

# Quantization imports
try:
    from sglang.srt.layers.quantization import Fp8Config
    print("✅ import quantization")
except Exception as e:
    errors.append(f"❌ import quantization: {e}")

# DeepSeek model imports (the main target!)
try:
    from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
    print("✅ import DeepSeek v2 model")
except Exception as e:
    errors.append(f"❌ import DeepSeek v2: {e}")

try:
    from sglang.srt.models.deepseek_common import MLA
    print("✅ import DeepSeek MLA attention")
except Exception as e:
    errors.append(f"❌ import MLA: {e}")

# Model executor (where platform code was heavily removed)
try:
    from sglang.srt.model_executor.model_runner import ModelRunner
    print("✅ import ModelRunner")
except Exception as e:
    errors.append(f"❌ import ModelRunner: {e}")

# Distributed (platform code removed)
try:
    from sglang.srt.distributed import initialize_distributed
    print("✅ import distributed")
except Exception as e:
    errors.append(f"❌ import distributed: {e}")

# Memory management (platform code removed)
try:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    print("✅ import memory pool")
except Exception as e:
    errors.append(f"❌ import memory pool: {e}")

# Summary
print("\n" + "="*60)
if errors:
    print(f"❌ FAILED: {len(errors)} import errors found:")
    for err in errors:
        print(f"   {err}")
    sys.exit(1)
else:
    print("✅ SUCCESS: All imports passed!")
    print("   No platform-specific errors detected")
    sys.exit(0)
EOF
```

## Phase 6: Check for Platform References (Validation)

```bash
# This should return 0 platform references in non-test code
cd /path/to/mini-sglang-deepseek-only/sglang/python

grep -r "is_npu\|is_xpu\|is_hip" --include="*.py" sglang/srt/ | grep -v test | wc -l
# Expected: 0

# If you find any, they should only be in:
# 1. test files (sglang/test/*.py) - acceptable
# 2. __init__.py exports - acceptable if not used
# 3. Comments - acceptable

# Check specific critical files
echo "Checking critical files for platform references..."
for file in \
    "sglang/srt/layers/activation.py" \
    "sglang/srt/layers/layernorm.py" \
    "sglang/srt/model_executor/model_runner.py" \
    "sglang/srt/layers/moe/topk.py"; do

    count=$(grep -c "is_npu\|is_xpu\|is_hip" "$file" 2>/dev/null || echo 0)
    if [ "$count" -eq 0 ]; then
        echo "✅ $file - clean"
    else
        echo "❌ $file - $count platform references found"
    fi
done
```

## Phase 7: Quick CUDA Kernel Test

```bash
# Test a simple kernel operation
python3 << 'EOF'
import torch
import sgl_kernel

print("Testing CUDA kernel functionality...")

# Test device
if not torch.cuda.is_available():
    print("❌ CUDA not available")
    exit(1)

device = torch.device("cuda")
print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")

# Test simple kernel: rmsnorm (used by DeepSeek)
hidden_size = 4096
batch_size = 2
eps = 1e-6

x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
out = torch.empty_like(x)

# Call rmsnorm kernel
sgl_kernel.rmsnorm(x, weight, eps)

print(f"✅ rmsnorm kernel executed successfully")
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {out.shape}")

# Test MoE kernel: moe_fused_gate
try:
    # Basic test - just check if function exists
    assert hasattr(sgl_kernel, 'moe_fused_gate')
    print("✅ moe_fused_gate kernel available (MoE routing)")
except Exception as e:
    print(f"⚠️  moe_fused_gate test skipped: {e}")

print("\n✅ CUDA kernel test passed!")
EOF
```

## Common Issues and Solutions

### Issue 1: CUDA Toolkit Not Found
```
Error: Could not find CUDA toolkit
```

**Solution:**
```bash
# Check CUDA installation
nvcc --version
which nvcc

# If not found, install CUDA toolkit
# For Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue 2: Platform Import Errors
```
ImportError: cannot import name 'is_npu' from 'sglang.srt.utils'
```

**Solution:**
This means cleanup was incomplete. The file should not reference platform functions.
```bash
# Find the problematic file
grep -rn "is_npu" sglang/python/sglang/srt/

# Report back which file has the issue - we may have missed one
```

### Issue 3: Missing PyTorch CUDA
```
Error: torch not compiled with CUDA enabled
```

**Solution:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: CMake Version Too Old
```
Error: CMake 3.18 or higher is required
```

**Solution:**
```bash
pip install cmake --upgrade
```

## Success Criteria

✅ **Phase 2:** sgl-kernel compiles without platform errors
✅ **Phase 3:** sgl_kernel module imports successfully
✅ **Phase 4:** sglang package installs without platform dependencies
✅ **Phase 5:** All sglang modules import without platform errors
✅ **Phase 6:** Zero platform references found in non-test code
✅ **Phase 7:** CUDA kernels execute on GPU

## Expected Timeline

- **Setup & Clone:** 5 minutes
- **sgl-kernel Compilation:** 10-20 minutes (depending on GPU/CPU)
- **sglang Installation:** 5-10 minutes
- **Import Testing:** 2 minutes
- **CUDA Kernel Testing:** 2 minutes

**Total:** ~30-40 minutes

## Next Steps After Successful Compilation

Once sgl-kernel compiles successfully:

1. **Run full validation:** `python3 scripts/validate_on_mac.py`
2. **Test DeepSeek model loading:** Try loading DeepSeek config
3. **Single layer test:** Test 1 standard layer + 1 MoE layer
4. **Benchmark scripts:** Use `benchmark/deepseek_v3/*.py` scripts

## Reporting Back

Please report:
1. ✅ **Success:** "sgl-kernel compiled successfully, all imports work"
2. ⚠️ **Warnings:** Any compilation warnings (usually safe to ignore)
3. ❌ **Errors:** Full error messages if compilation fails
4. 📊 **System info:**
   - GPU model: `nvidia-smi --query-gpu=name --format=csv,noheader`
   - CUDA version: `nvcc --version`
   - PyTorch CUDA: `python3 -c "import torch; print(torch.version.cuda)"`

---

**Good luck with the single-node GPU testing!** 🚀

Let me know how the sgl-kernel compilation goes!
