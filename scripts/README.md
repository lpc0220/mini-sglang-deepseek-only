# Scripts Directory

This directory contains scripts used for codebase cleanup, validation, and testing.

## Active Scripts (Use These)

### validate_on_mac.py
**Purpose:** Comprehensive validation suite for Mac/Linux environment
**Usage:** `python3 validate_on_mac.py`

**Tests performed:**
1. Python import validation
2. Platform-specific code checks
3. DeepSeek model constants validation
4. Build configuration verification
5. Test discovery validation
6. Documentation validation
7. Git status check

See [docs/MAC_VALIDATION_REPORT.md](../docs/MAC_VALIDATION_REPORT.md) for results.

## Historical Scripts (Already Executed - DO NOT RE-RUN)

### platform_cleanup/
**Purpose:** Scripts used to remove platform-specific code (NPU, XPU, HIP/AMD, CPU)

**Status:** ✅ Cleanup complete - 44 files cleaned

**Scripts:**
- Early phase: remove_npu_refs.py, remove_hip_conditionals.py, remove_cpu_refs.py
- Import cleanup: cleanup_remaining_platform_imports.py (4 iterations)
- Final cleanup: cleanup_all_platform_imports_comprehensive.py

See [platform_cleanup/README.md](platform_cleanup/README.md) for details.

### cleanup_phase3c/
**Purpose:** Phase 3C platform code removal (earlier cleanup phase)

**Status:** ✅ Cleanup complete - 59 files modified

**Scripts:**
- HIP/ROCm cleanup: clean_hip_*.py
- Platform removal: phase3c_remove_xpu.py, phase3c_complete_cleanup.py
- Validation: validate_phase3c.py

**Results:**
- 7,333 lines removed
- 100% platform conditionals removed
- CUDA-only codebase achieved

### archive/
**Purpose:** Archived scripts from earlier phases

## Directory Structure

```
scripts/
├── README.md                           # This file
├── validate_on_mac.py                 # Active: Validation suite
├── platform_cleanup/                  # Historical: Platform import cleanup
│   ├── README.md
│   └── 9 cleanup scripts
├── cleanup_phase3c/                   # Historical: Phase 3C cleanup
│   └── 10 cleanup/validation scripts
└── archive/                           # Historical: Archived scripts
```

## Usage Notes

### For GPU Cluster Testing
On the GPU cluster, you only need to run:
```bash
python3 validate_on_mac.py
```

This will verify:
- ✅ All imports work correctly
- ✅ No platform-specific code remains
- ✅ Build configurations are CUDA-only
- ✅ Test structure is intact

### DO NOT Re-run Historical Scripts
All cleanup scripts have already been executed and their changes are committed to the repository. Re-running them may cause errors or unexpected behavior.

## Cleanup Results Summary

**Total cleanup across all phases:**
- **Files modified:** 103+ files
- **Lines removed:** ~377,000 lines (56% reduction)
- **Platform code:** 100% removed (NPU, XPU, HIP, CPU-specific)
- **Build configs:** 6 platform configs deleted
- **Test files:** 15 non-DeepSeek tests removed
- **Platform imports:** 56+ import statements removed from 44 files

**Final state:** 100% NVIDIA CUDA-only codebase for DeepSeek models

## Related Documentation

- [docs/MAC_VALIDATION_REPORT.md](../docs/MAC_VALIDATION_REPORT.md) - Mac validation results
- [docs/BUILD_SYSTEM_UPDATE.md](../docs/BUILD_SYSTEM_UPDATE.md) - Build system changes
- [docs/TEST_CLEANUP_COMPLETE.md](../docs/TEST_CLEANUP_COMPLETE.md) - Test cleanup results
- [docs/READY_FOR_GPU_TESTING.md](../docs/READY_FOR_GPU_TESTING.md) - GPU testing guide
- [docs/SINGLE_NODE_GPU_TESTING.md](../docs/SINGLE_NODE_GPU_TESTING.md) - Single-node GPU guide
