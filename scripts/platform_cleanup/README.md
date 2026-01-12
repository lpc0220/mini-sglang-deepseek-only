# Platform Cleanup Scripts

**Purpose:** Historical scripts used to remove platform-specific code (NPU, XPU, HIP/AMD, CPU) from the codebase.

**Status:** ✅ Cleanup complete - these scripts were already executed successfully.

## Scripts (Execution Order)

### Phase 1: Early Platform Reference Removal
1. **remove_npu_refs.py** - Remove NPU/Ascend platform references
2. **remove_hip_conditionals.py** - Remove HIP/AMD ROCm conditionals
3. **remove_cpu_refs.py** - Remove CPU-only backend references
4. **cleanup_unused_imports.py** - Clean up unused imports after removal
5. **final_platform_cleanup.py** - Final cleanup pass

### Phase 2: Import Statement Cleanup
6. **cleanup_remaining_platform_imports.py** - First pass (7 files)
7. **cleanup_all_remaining_platform_imports.py** - Second pass (8 files)
8. **cleanup_final_platform_imports.py** - Third pass (9 files)
9. **cleanup_all_platform_imports_comprehensive.py** - Final pass (14 files)

**Total:** 44 files cleaned across all phases

## Results

✅ **All platform imports removed:**
- `is_npu` (Ascend NPU) - REMOVED
- `is_xpu` (Intel XPU) - REMOVED
- `is_hip` (AMD ROCm/HIP) - REMOVED
- `is_cpu` (CPU-only backend) - REMOVED

✅ **Validation:** Zero platform references remain in non-test code

## Files Modified (44 total)

See [docs/MAC_VALIDATION_REPORT.md](../../docs/MAC_VALIDATION_REPORT.md) for complete list.

## Do NOT Re-run These Scripts

These scripts have already been executed and their changes are committed to the repository. Re-running them may cause errors or unexpected behavior.

If you need to verify the cleanup was successful, use:
```bash
cd /path/to/sglang/python
grep -r "is_npu\|is_xpu\|is_hip" --include="*.py" sglang/srt/ | grep -v test | wc -l
# Should output: 0
```

## Related Documentation

- [docs/MAC_VALIDATION_REPORT.md](../../docs/MAC_VALIDATION_REPORT.md) - Complete validation results
- [docs/BUILD_SYSTEM_UPDATE.md](../../docs/BUILD_SYSTEM_UPDATE.md) - Build system changes
- [docs/TEST_CLEANUP_COMPLETE.md](../../docs/TEST_CLEANUP_COMPLETE.md) - Test cleanup results
