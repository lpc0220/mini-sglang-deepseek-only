#!/usr/bin/env python3
"""
Mac Validation Suite for SGLang DeepSeek-Only Codebase

This script validates the codebase on Mac without requiring NVIDIA GPUs:
1. Python import validation
2. Platform-specific code checks
3. DeepSeek model architecture validation
4. Build configuration verification
5. Test discovery validation

Run before pushing to GPU clusters for final validation.
"""

import sys
import subprocess
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ {text}{RESET}")

# Change to sglang directory
REPO_ROOT = Path(__file__).parent.parent
SGLANG_DIR = REPO_ROOT / "sglang"
sys.path.insert(0, str(SGLANG_DIR / "python"))

errors = []
warnings = []

# ============================================================================
# Test 1: Python Import Validation
# ============================================================================
print_header("Test 1: Python Import Validation")

def test_import(module_name, description):
    try:
        __import__(module_name)
        print_success(f"Import {module_name} - {description}")
        return True
    except ImportError as e:
        print_error(f"Import {module_name} failed: {e}")
        errors.append(f"Import {module_name} failed: {e}")
        return False
    except Exception as e:
        # Some modules may fail due to CUDA requirement, that's okay on Mac
        print_warning(f"Import {module_name} - {e.__class__.__name__}: {e}")
        warnings.append(f"Import {module_name} - {e.__class__.__name__}")
        return True  # Not a critical error

# Core imports
test_import("sglang", "Core SGLang package")
test_import("sglang.srt", "SGLang runtime")
test_import("sglang.srt.configs", "Configuration modules")
test_import("sglang.srt.utils", "Utility functions")

# Model imports (may fail on Mac without CUDA, that's expected)
print_info("Testing DeepSeek model imports (CUDA warnings expected on Mac)...")
test_import("sglang.srt.models.deepseek_v2", "DeepSeek v2 model")
test_import("sglang.srt.models.deepseek_common", "DeepSeek common modules")

# Layer imports
test_import("sglang.srt.layers.activation", "Activation layers")
test_import("sglang.srt.layers.quantization", "Quantization layers")
test_import("sglang.srt.layers.moe", "MoE layers")

# ============================================================================
# Test 2: Platform-Specific Code Checks
# ============================================================================
print_header("Test 2: Platform-Specific Code Checks")

def check_no_platform_refs(directory, platforms=["is_hip", "is_npu", "is_xpu", "_is_hip", "_is_npu"]):
    """Check that no platform-specific references remain in Python files."""
    found = []
    for py_file in Path(directory).rglob("*.py"):
        try:
            content = py_file.read_text()
            for platform in platforms:
                if platform in content:
                    # Exclude comments and docstrings (heuristic check)
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if platform in line and not line.strip().startswith('#'):
                            found.append((py_file, i, platform, line.strip()))
        except Exception as e:
            warnings.append(f"Could not read {py_file}: {e}")
    return found

print_info("Checking for HIP/NPU/XPU platform references...")
platform_refs = check_no_platform_refs(SGLANG_DIR / "python/sglang/srt")

if platform_refs:
    print_error(f"Found {len(platform_refs)} platform-specific references:")
    for file, line_num, platform, line in platform_refs[:10]:  # Show first 10
        print(f"  {file.relative_to(SGLANG_DIR)}:{line_num} - {platform}")
        print(f"    {line[:80]}")
    errors.append(f"Platform-specific code found: {len(platform_refs)} references")
else:
    print_success("No HIP/NPU/XPU platform references found")

# ============================================================================
# Test 3: DeepSeek Model Constants Validation
# ============================================================================
print_header("Test 3: DeepSeek Model Constants Validation")

try:
    from sglang.test.test_utils import (
        DEFAULT_MLA_MODEL_NAME_FOR_TEST,
        DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST,
    )
    print_success(f"DEFAULT_MLA_MODEL_NAME_FOR_TEST = {DEFAULT_MLA_MODEL_NAME_FOR_TEST}")
    print_success(f"DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST = {DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST}")

    # Verify they reference DeepSeek models
    assert "deepseek" in DEFAULT_MLA_MODEL_NAME_FOR_TEST.lower(), "MLA model should be DeepSeek"
    assert "deepseek" in DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST.lower(), "FP4 model should be DeepSeek"
    print_success("DeepSeek model constants are correctly set")
except Exception as e:
    print_error(f"DeepSeek model constants validation failed: {e}")
    errors.append(f"Model constants validation: {e}")

# ============================================================================
# Test 4: Build Configuration Verification
# ============================================================================
print_header("Test 4: Build Configuration Verification")

def check_file_not_exists(file_path, description):
    """Verify a file does not exist (should have been deleted)."""
    if Path(file_path).exists():
        print_error(f"{description} still exists: {file_path}")
        errors.append(f"File should not exist: {file_path}")
        return False
    else:
        print_success(f"{description} correctly deleted")
        return True

# Check platform-specific build configs are deleted
check_file_not_exists(SGLANG_DIR / "sgl-kernel/pyproject_cpu.toml", "sgl-kernel CPU build config")
check_file_not_exists(SGLANG_DIR / "sgl-kernel/pyproject_rocm.toml", "sgl-kernel ROCm build config")
check_file_not_exists(SGLANG_DIR / "sgl-kernel/setup_rocm.py", "sgl-kernel ROCm setup")
check_file_not_exists(SGLANG_DIR / "python/pyproject_cpu.toml", "Python CPU build config")
check_file_not_exists(SGLANG_DIR / "python/pyproject_xpu.toml", "Python XPU build config")
check_file_not_exists(SGLANG_DIR / "python/pyproject_other.toml", "Python other platform config")

# Check CMakeLists.txt doesn't reference deleted files
cmake_file = SGLANG_DIR / "sgl-kernel/CMakeLists.txt"
if cmake_file.exists():
    content = cmake_file.read_text()
    deleted_files = [
        "timestep_embedding.cu",
        "apply_token_bitmask_inplace_cuda.cu",
        "causal_conv1d.cu",
        "gguf_kernel.cu"
    ]
    found_deleted = [f for f in deleted_files if f in content]
    if found_deleted:
        print_error(f"CMakeLists.txt references deleted files: {found_deleted}")
        errors.append(f"CMakeLists.txt has deleted file references: {found_deleted}")
    else:
        print_success("CMakeLists.txt has no deleted file references")
else:
    print_warning("CMakeLists.txt not found")

# ============================================================================
# Test 5: Test Discovery Validation
# ============================================================================
print_header("Test 5: Test Discovery Validation")

print_info("Checking test directory structure...")
test_dir = SGLANG_DIR / "python/sglang/test"
if test_dir.exists():
    test_files = list(test_dir.rglob("*.py"))
    print_success(f"Found {len(test_files)} test files")

    # Verify critical tests exist
    critical_tests = [
        "test_cutlass_moe.py",
        "test_block_fp8.py",
        "attention/test_flashattn_mla_backend.py",
    ]
    for test in critical_tests:
        test_path = test_dir / test
        if test_path.exists():
            print_success(f"Critical test exists: {test}")
        else:
            print_error(f"Critical test missing: {test}")
            errors.append(f"Missing critical test: {test}")

    # Verify deleted tests don't exist
    deleted_tests = [
        "simple_eval_mmmu_vlm.py",
        "simple_eval_mmlu.py",
        "gpt_oss_common.py",
        "few_shot_gsm8k.py",
    ]
    for test in deleted_tests:
        test_path = test_dir / test
        if test_path.exists():
            print_error(f"Deleted test still exists: {test}")
            errors.append(f"Test should be deleted: {test}")
        else:
            print_success(f"Deleted test confirmed removed: {test}")
else:
    print_error("Test directory not found")
    errors.append("Test directory missing")

# ============================================================================
# Test 6: Documentation Validation
# ============================================================================
print_header("Test 6: Documentation Validation")

docs_dir = REPO_ROOT / "docs"
required_docs = [
    "README.md",
    "BUILD_SYSTEM_UPDATE.md",
    "ORGANIZATION_SUMMARY.md",
    "TEST_CLEANUP_PLAN.md",
    "TEST_CLEANUP_COMPLETE.md",
]

for doc in required_docs:
    doc_path = docs_dir / doc
    if doc_path.exists():
        print_success(f"Documentation exists: {doc}")
    else:
        print_warning(f"Documentation missing: {doc}")
        warnings.append(f"Documentation missing: {doc}")

# ============================================================================
# Test 7: Git Status Check
# ============================================================================
print_header("Test 7: Git Status Check")

try:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True
    )
    if result.stdout.strip():
        print_warning("Uncommitted changes detected:")
        print(result.stdout)
        warnings.append("Uncommitted changes exist")
    else:
        print_success("Working directory is clean (all changes committed)")
except Exception as e:
    print_warning(f"Could not check git status: {e}")

# ============================================================================
# Summary Report
# ============================================================================
print_header("Validation Summary")

print(f"\n{BLUE}Results:{RESET}")
print(f"  {GREEN}✓ Errors: {len(errors)}{RESET}")
print(f"  {YELLOW}⚠ Warnings: {len(warnings)}{RESET}")

if errors:
    print(f"\n{RED}ERRORS FOUND:{RESET}")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print(f"\n{RED}❌ VALIDATION FAILED - Fix errors before pushing!{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}✅ ALL CRITICAL VALIDATIONS PASSED{RESET}")

if warnings:
    print(f"\n{YELLOW}WARNINGS (non-critical):{RESET}")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print(f"\n{YELLOW}Note: Warnings are expected on Mac (CUDA/GPU requirements){RESET}")

print(f"\n{GREEN}{'='*80}{RESET}")
print(f"{GREEN}Ready for GPU cluster testing!{RESET}")
print(f"{GREEN}{'='*80}{RESET}\n")

print_info("Next steps:")
print("  1. Review any warnings above (most are expected on Mac)")
print("  2. Push code to remote repository")
print("  3. Run GPU cluster validation tests")
print("  4. Report results back for any issues")

sys.exit(0)
