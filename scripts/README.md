# Cleanup Scripts

This directory contains automated cleanup scripts used during the SGLang shrinking project.

## Directory Structure

### `/cleanup_phase3c/`
Scripts used during Phase 3C (Platform Code Removal) and related cleanup phases:

#### HIP/ROCm Cleanup Scripts
- **clean_hip_attention.py** - Remove HIP conditionals from attention modules
- **clean_hip_quantization.py** - Remove HIP conditionals from quantization code
- **clean_hip_server_args.py** - Remove HIP references from server arguments
- **cleanup_hip_refs.py** - General HIP/ROCm reference removal

#### Platform Removal Scripts
- **phase3c_remove_xpu.py** - Remove Intel XPU platform code
- **phase3c_complete_cleanup.py** - Comprehensive platform cleanup (NPU/HIP/XPU)
- **phase3c_final_cleanup.py** - Final pass platform cleanup
- **phase3c_ultra_final_cleanup.py** - Ultimate cleanup pass

#### Validation Scripts
- **validate_phase3c.py** - Validate Phase 3C cleanup results
- **validate_phase3c_final.py** - Final validation of platform removal

#### Status Files
- **PHASE3C_COMPLETE.txt** - Phase 3C completion marker

## Usage Notes

These scripts were used during the active cleanup phases and are preserved for:
1. **Documentation** - Understanding what was removed and how
2. **Reference** - Template for future similar cleanup operations
3. **Audit Trail** - Complete record of automated transformations

**DO NOT RUN THESE SCRIPTS** - They have already been executed and their changes are committed. Running them again may cause errors or unexpected behavior.

## What Was Cleaned Up

### Phase 3C Focus: Platform-Specific Code Removal
- **AMD/HIP/ROCm:** All AMD GPU platform code and conditionals
- **Ascend NPU:** All NPU backend code and conditionals
- **Intel XPU:** All XPU platform references
- **CPU-only paths:** Platform-specific CPU fallback code

### What Was Preserved
✅ **NVIDIA CUDA code** - All CUDA kernels and GPU operations
✅ **Quantization** - All quantization methods (FP4, FP8, AWQ, GPTQ, etc.)
✅ **DeepSeek models** - All DeepSeek model implementations
✅ **CPU kernels for Mac testing** - CPU operations needed for testing

## Results

- **Files modified:** 59 files (via automated cleanup)
- **Lines removed:** ~7,333 lines of platform-specific code
- **Platform conditionals removed:** 100% (NPU/HIP/XPU)
- **Final state:** 100% NVIDIA CUDA-only codebase

See [docs/phase_reports/](../docs/phase_reports/) for detailed completion reports.
