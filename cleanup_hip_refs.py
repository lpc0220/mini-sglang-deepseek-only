#!/usr/bin/env python3
"""
Batch cleanup of HIP/ROCm/AMD references from SGLang codebase.
Removes conditional branches while preserving all CUDA implementations.
"""

import re
from pathlib import Path

# Files to process
FILES_TO_CLEAN = [
    # MoE layer files (remaining)
    "sglang/python/sglang/srt/layers/moe/ep_moe/layer.py",
    "sglang/python/sglang/srt/layers/moe/moe_runner/triton.py",
    "sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py",
    "sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py",
    "sglang/python/sglang/srt/layers/moe/token_dispatcher/standard.py",
    "sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py",
    # Quantization files
    "sglang/python/sglang/srt/layers/quantization/fp8.py",
    "sglang/python/sglang/srt/layers/quantization/fp8_kernel.py",
    "sglang/python/sglang/srt/layers/quantization/fp8_utils.py",
    "sglang/python/sglang/srt/layers/quantization/mxfp4.py",
    "sglang/python/sglang/srt/layers/quantization/awq.py",
    "sglang/python/sglang/srt/layers/quantization/gguf.py",
    "sglang/python/sglang/srt/layers/quantization/unquant.py",
    "sglang/python/sglang/srt/layers/quantization/__init__.py",
    "sglang/python/sglang/srt/layers/quantization/petit.py",
    "sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py",
    "sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py",
    "sglang/python/sglang/srt/layers/quantization/quark/quark_moe.py",
    "sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py",
    "sglang/python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
    # Attention files
    "sglang/python/sglang/srt/layers/attention/nsa_backend.py",
    "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py",
    "sglang/python/sglang/srt/layers/attention/nsa/tilelang_kernel.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py",
    "sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py",
    "sglang/python/sglang/srt/layers/attention/merge_state.py",
    # Server args
    "sglang/python/sglang/srt/server_args.py",
]


def remove_hip_imports(content: str) -> str:
    """Remove HIP-related imports."""
    # Remove is_hip import
    content = re.sub(
        r',\s*is_hip\s*(?=\)|,)', '', content
    )
    content = re.sub(
        r'is_hip\s*,\s*', '', content
    )

    return content


def remove_hip_module_vars(content: str) -> str:
    """Remove HIP module-level variables."""
    # Remove _is_hip = is_hip()
    content = re.sub(
        r'_is_hip\s*=\s*is_hip\(\)\s*\n', '', content
    )

    # Remove _use_aiter = ...
    content = re.sub(
        r'_use_aiter\s*=\s*.*?_is_hip.*?\n', '', content
    )
    content = re.sub(
        r'_use_aiter\s*=\s*.*?is_hip\(\).*?\n', '', content
    )

    return content


def simplify_cuda_hip_or(content: str) -> str:
    """Simplify `if _is_cuda or _is_hip:` to `if _is_cuda:`."""
    content = re.sub(
        r'if\s+_is_cuda\s+or\s+_is_hip\s*:', 'if _is_cuda:', content
    )
    content = re.sub(
        r'elif\s+_is_cuda\s+or\s+_is_hip\s*:', 'elif _is_cuda:', content
    )

    return content


def clean_file(file_path: Path) -> tuple[bool, str]:
    """Clean a single file of HIP references.

    Returns:
        (success, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        content = file_path.read_text()
        original_lines = content.count('\n')

        # Apply transformations
        content = remove_hip_imports(content)
        content = remove_hip_module_vars(content)
        content = simplify_cuda_hip_or(content)

        # Write back
        file_path.write_text(content)

        new_lines = content.count('\n')
        lines_removed = original_lines - new_lines

        return True, f"Removed {lines_removed} lines"

    except Exception as e:
        return False, f"Error: {e}"


def main():
    base_path = Path("/Users/lpc/workspace/sglang-deepseek-only")

    print("Phase 3C: Cleaning HIP references from remaining files...")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for file_rel in FILES_TO_CLEAN:
        file_path = base_path / file_rel
        success, msg = clean_file(file_path)

        status = "✅" if success else "❌"
        print(f"{status} {file_rel}: {msg}")

        if success:
            success_count += 1
        else:
            fail_count += 1

    print("=" * 70)
    print(f"Summary: {success_count} files cleaned, {fail_count} files failed")


if __name__ == "__main__":
    main()
