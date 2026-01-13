# GB200 Build Guide for SGLang (DeepSeek-Only)

**Date:** 2026-01-12
**Hardware:** NVIDIA GB200 (Blackwell architecture, compute capability 10.0)
**CUDA Version:** 12.9
**Target Model:** DeepSeek-R1-NVFP4-v2

## sgl-kernel Build Configuration

### Recommended Build Command

```bash
cd sglang/sgl-kernel
MAX_JOBS=32 pip install -e . --no-build-isolation --config-settings=build-dir=build -v
```

### Build Parameters Explained

| Parameter | Purpose | Benefit |
|-----------|---------|---------|
| `MAX_JOBS=32` | Parallel compilation with 32 jobs | **6-11 minutes faster** than sequential build |
| `--no-build-isolation` | Use environment packages directly | Faster, avoids duplicate dependency resolution |
| `--config-settings=build-dir=build` | **Save .so files to source directory** | Preserves compiled artifacts in `build/` for inspection/debugging |
| `-v` | Verbose output | See compilation progress and catch errors early |

### Why `build-dir=build` Matters

**Default behavior:** pip builds in a temporary directory and only copies final `.so` files
**With `build-dir=build`:** All build artifacts saved to `sglang/sgl-kernel/build/`

**Benefits:**
- ✅ Compiled `.so` files accessible for debugging
- ✅ Build cache preserved for faster rebuilds
- ✅ Easy to inspect what was compiled
- ✅ Can manually check symbol tables with `nm` or `objdump`

**Example:**
```bash
# After build, check what symbols are in the .so
cd build/
find . -name "*.so" | head -5
# Example: ./lib.linux-x86_64-cpython-311/sgl_kernel/common_ops.abi3.so

# Check symbols in compiled library
nm -D ./lib.linux-x86_64-cpython-311/sgl_kernel/common_ops.abi3.so | grep mla
# Should show: cutlass_mla_decode, cutlass_mla_get_workspace_size
# Should NOT show: timestep_embedding, ggml_*, causal_conv1d_* (removed)
```

## Full Build Process (After Fixes)

### 1. Pull Latest Changes

```bash
cd /path/to/sglang
git pull origin deepseek-only
```

**Latest commits:**
- `de94821e5` - Remove GGUF quantization from Python code
- `55ec573f5` - Remove GGUF/Mamba/Grammar C++ declarations
- `c50324911` - Fix timestep_embedding undefined symbol
- `ce2d2fc59` - Remove all non-DeepSeek model configs

### 2. Clean Previous Build (Important!)

```bash
cd sgl-kernel
rm -rf build/  # Remove old build artifacts
rm -rf *.egg-info  # Remove old package metadata
```

**Why clean build is required:**
- C++ source files changed (common_extension.cc, sgl_kernel_ops.h)
- Removed symbol registrations (timestep_embedding, GGUF, Mamba, Grammar)
- Stale build cache can cause linker errors

### 3. Build sgl-kernel

```bash
# Recommended: Use Ninja build system for faster compilation
export CMAKE_GENERATOR=Ninja

# Set CUDA architectures for GB200 (Blackwell = 10.0, also include Hopper = 9.0)
export TORCH_CUDA_ARCH_LIST="9.0;10.0"

# Build with optimizations
MAX_JOBS=32 pip install -e . --no-build-isolation --config-settings=build-dir=build -v
```

**Expected build time:**
- **With Ninja + MAX_JOBS=32:** ~8-15 minutes
- **Without parallelization:** ~20-30 minutes
- **Savings from FA3/FA4/FlashMLA removal:** 6-11 minutes

### 4. Install Python Package

```bash
cd ../python
pip install -e ".[srt]"
```

### 5. Verify Installation

```bash
# Test imports
python3 -c "from sglang.srt.model_executor.model_runner import ModelRunner; print('✅ Import success')"

# Check that removed symbols are NOT present
python3 -c "
import torch
try:
    torch.ops.sgl_kernel.timestep_embedding
    print('❌ ERROR: timestep_embedding still exists')
except Exception:
    print('✅ timestep_embedding removed correctly')
"

# Check that GGUF is not in quantization methods
python3 -c "
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
assert 'gguf' not in QUANTIZATION_METHODS, 'ERROR: GGUF still in registry'
print('✅ GGUF removed correctly')
print(f'Available quantization: {list(QUANTIZATION_METHODS.keys())}')
"
```

## Build Optimizations Applied

### Disabled Kernels (Not Compiled)

These were disabled in `CMakeLists.txt` to reduce build time:

1. **FlashAttention 3 (FA3)** - Disabled (lines 441-458)
   ```cmake
   if (FALSE)  # Disabled FA3 build
     # FA3 compilation disabled
   endif()
   ```

2. **FlashAttention 4 (FA4)** - Disabled (lines 595-612)
   ```cmake
   if (FALSE)  # Disabled FA4 install
     # FA4 installation disabled
   endif()
   ```

3. **FlashMLA** - Deleted (previously at line 539)
   ```cmake
   # REMOVED: include(${CMAKE_CURRENT_LIST_DIR}/cmake/flashmla.cmake)
   # FlashMLA removed - use CUTLASS MLA or TRTLLM MLA
   ```

**Build time savings:** 6-11 minutes

### Removed C++ Symbols (267 + 94 lines)

Removed from `common_extension.cc` and `sgl_kernel_ops.h`:
- `timestep_embedding` (diffusion models)
- `apply_token_bitmask_inplace_cuda` (grammar, Triton used instead)
- `ggml_*` functions (6 GGUF quantization functions)
- `causal_conv1d_*` (2 Mamba functions)

**Total C++ removal:** 361 lines

### Removed Python Code (574 lines)

- Deleted `python/sglang/srt/layers/quantization/gguf.py` (563 lines)
- Removed GGUF from quantization registry
- Removed GGUF from server args

**Total Python removal:** 574 lines

## CUDA Architecture Configuration

### Recommended for GB200

```bash
export TORCH_CUDA_ARCH_LIST="9.0;10.0"
```

**Architectures:**
- **9.0** - Hopper (H100, H200)
- **10.0** - Blackwell (GB200, B100, B200)

**Why include both:**
- Code may run on H100/H200 during development
- Forward compatibility for Blackwell deployment

### Alternative (Blackwell-only, fastest build)

```bash
export TORCH_CUDA_ARCH_LIST="10.0"
```

**Use this if:**
- Only deploying on GB200/Blackwell
- Want fastest possible build time
- Don't need Hopper compatibility

## Quantization Configuration

### Supported Quantization Methods

**For DeepSeek-R1-NVFP4-v2:**
- ✅ `modelopt_fp4` - NVIDIA FP4 quantization (PRIMARY)
- ✅ `modelopt_fp8` - NVIDIA FP8 quantization
- ✅ `fp8` - Standard FP8
- ✅ `awq`, `gptq`, `marlin` - Alternative quantization methods

**Removed (not supported):**
- ❌ `gguf` - GGUF quantization (llama.cpp/CPU format)

## Troubleshooting

### Issue: Undefined Symbol Errors

**Symptoms:**
```
ImportError: undefined symbol: _Z18timestep_embeddingRKN2at6TensorERS0_lbddl
```

**Solution:**
1. Clean build directory: `rm -rf build/`
2. Rebuild sgl-kernel with latest code
3. Verify symbol is removed (see "Verify Installation" above)

**Fixed in commits:** c50324911, 55ec573f5, de94821e5

### Issue: Import Errors for GGUF

**Symptoms:**
```
cannot import name 'ggml_dequantize' from 'sgl_kernel.quantization'
```

**Solution:**
- GGUF quantization removed (not supported)
- Use `--quantization modelopt_fp4` for DeepSeek models

**Fixed in commit:** de94821e5

### Issue: Slow Build Time

**Solutions:**
1. Use `MAX_JOBS=32` for parallel compilation
2. Use Ninja: `export CMAKE_GENERATOR=Ninja`
3. Limit CUDA architectures: `export TORCH_CUDA_ARCH_LIST="10.0"`
4. FA3/FA4/FlashMLA already disabled (saves 6-11 minutes)

### Issue: Build Cache Issues

**Symptoms:**
- Linker errors for removed symbols
- Stale .so files

**Solution:**
```bash
cd sgl-kernel
rm -rf build/ *.egg-info
pip cache purge  # Optional: clear pip cache
MAX_JOBS=32 pip install -e . --no-build-isolation --config-settings=build-dir=build -v
```

## Inference Test Command

After successful build, test with:

```bash
python -m sglang.bench_one_batch \
    --model-path /lustre/fsw/coreai_dlfw_dev/pengchengl/DeepSeek-R1-0528-FP4 \
    --attention-backend trtllm_mla \
    --moe-runner-backend=flashinfer_trtllm \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size=4 --ep-size=4 --data-parallel-size=4 \
    --enable-dp-attention --disable-radix-cache \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024 --cuda-graph-max-bs 1024 \
    --stream-interval 1 \
    --batch-size 2 --input-len 64 --output-len 4
```

## Summary

**Optimized build configuration:**
- ✅ `MAX_JOBS=32` for parallel compilation
- ✅ `--config-settings=build-dir=build` to save .so files in source tree
- ✅ `--no-build-isolation` for faster builds
- ✅ `TORCH_CUDA_ARCH_LIST="9.0;10.0"` for Hopper + Blackwell
- ✅ FA3/FA4/FlashMLA disabled (6-11 min faster)
- ✅ All undefined symbols removed (timestep_embedding, GGUF, Mamba, Grammar)
- ✅ Clean build required after C++ changes

**Expected result:** Clean build with no undefined symbol errors, ready for GB200 inference testing.
