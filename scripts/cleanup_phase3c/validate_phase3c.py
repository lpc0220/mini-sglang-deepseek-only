#!/usr/bin/env python3
"""
Validate Phase 3C cleanup:
1. Syntax check all cleaned files
2. Verify zero HIP references remain
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# All cleaned files (30 total: 8 MoE + 14 quant + 7 attention + 1 server_args)
CLEANED_FILES = [
    # MoE files (8)
    "sglang/python/sglang/srt/layers/moe/router.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
    "sglang/python/sglang/srt/layers/moe/moe_runner/triton.py",
    "sglang/python/sglang/srt/layers/moe/ep_moe/layer.py",
    "sglang/python/sglang/srt/layers/moe/topk.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py",
    "sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    "sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py",
    "sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py",
    # Quantization files (14)
    "sglang/python/sglang/srt/layers/quantization/gguf.py",
    "sglang/python/sglang/srt/layers/quantization/fp8.py",
    "sglang/python/sglang/srt/layers/quantization/unquant.py",
    "sglang/python/sglang/srt/layers/quantization/fp8_utils.py",
    "sglang/python/sglang/srt/layers/quantization/fp8_kernel.py",
    "sglang/python/sglang/srt/layers/quantization/__init__.py",
    "sglang/python/sglang/srt/layers/quantization/awq.py",
    "sglang/python/sglang/srt/layers/quantization/mxfp4.py",
    "sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
    "sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py",
    "sglang/python/sglang/srt/layers/quantization/quark/quark_moe.py",
    "sglang/python/sglang/srt/layers/quantization/petit.py",
    "sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py",
    "sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py",
    # Attention files (7)
    "sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py",
    "sglang/python/sglang/srt/layers/attention/nsa/tilelang_kernel.py",
    "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py",
    "sglang/python/sglang/srt/layers/attention/nsa_backend.py",
    # Server args (1)
    "sglang/python/sglang/srt/server_args.py",
]

BASE_DIR = Path("/Users/lpc/workspace/sglang-deepseek-only")

def syntax_check(filepath: Path) -> Tuple[bool, str]:
    """Check Python syntax using py_compile."""
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(filepath)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "OK"
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def check_hip_references(filepath: Path) -> Tuple[bool, List[str]]:
    """Check for remaining HIP references."""
    hip_patterns = [
        "_is_hip",
        "is_hip()",
        "_use_aiter",
        "from aiter",
        "import aiter",
    ]

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    found = []
    for i, line in enumerate(lines, 1):
        for pattern in hip_patterns:
            if pattern in line:
                found.append(f"Line {i}: {line.strip()}")

    return len(found) == 0, found

def main():
    print("=" * 80)
    print("PHASE 3C VALIDATION: AMD/ROCm/HIP Cleanup")
    print("=" * 80)
    print()

    # Step 1: Syntax validation
    print("Step 1: Syntax Validation")
    print("-" * 80)
    syntax_errors = []
    syntax_ok = 0

    for file_rel in CLEANED_FILES:
        filepath = BASE_DIR / file_rel
        if not filepath.exists():
            print(f"✗ NOT FOUND: {file_rel}")
            syntax_errors.append((file_rel, "File not found"))
            continue

        ok, msg = syntax_check(filepath)
        if ok:
            print(f"✓ {file_rel}")
            syntax_ok += 1
        else:
            print(f"✗ {file_rel}")
            print(f"  Error: {msg}")
            syntax_errors.append((file_rel, msg))

    print()
    print(f"Syntax Check: {syntax_ok}/{len(CLEANED_FILES)} files passed")
    print()

    # Step 2: HIP reference check
    print("Step 2: HIP Reference Check")
    print("-" * 80)
    hip_found = []
    hip_clean = 0

    for file_rel in CLEANED_FILES:
        filepath = BASE_DIR / file_rel
        if not filepath.exists():
            continue

        clean, refs = check_hip_references(filepath)
        if clean:
            print(f"✓ {file_rel} - No HIP references")
            hip_clean += 1
        else:
            print(f"✗ {file_rel} - HIP references found:")
            for ref in refs:
                print(f"    {ref}")
            hip_found.append((file_rel, refs))

    print()
    print(f"HIP Check: {hip_clean}/{len(CLEANED_FILES)} files clean")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files cleaned: {len(CLEANED_FILES)}")
    print(f"  - MoE layers: 12")
    print(f"  - Quantization: 14")
    print(f"  - Attention: 7")
    print(f"  - Server args: 1")
    print()
    print(f"Syntax validation: {syntax_ok}/{len(CLEANED_FILES)} passed")
    print(f"HIP reference check: {hip_clean}/{len(CLEANED_FILES)} clean")
    print()

    if syntax_errors:
        print("❌ SYNTAX ERRORS FOUND:")
        for file, error in syntax_errors:
            print(f"  - {file}: {error}")
        print()

    if hip_found:
        print("⚠️  HIP REFERENCES STILL FOUND:")
        for file, refs in hip_found:
            print(f"  - {file}: {len(refs)} references")
        print()

    if not syntax_errors and not hip_found:
        print("✅ ALL VALIDATIONS PASSED!")
        print("   - All files have valid syntax")
        print("   - Zero HIP references remain")
        print("   - Phase 3C complete!")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
