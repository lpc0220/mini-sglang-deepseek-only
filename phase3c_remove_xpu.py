#!/usr/bin/env python3
"""Remove all XPU references from SGLang."""

import re
from pathlib import Path

def clean_xpu_refs(file_path: Path) -> bool:
    """Remove XPU references from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Remove _is_xpu = is_xpu()
    content = re.sub(r'^\s*_is_xpu\s*=\s*is_xpu\(\)\s*$\n?', '', content, flags=re.MULTILINE)

    # if _is_cuda or _is_xpu: -> if _is_cuda:
    content = re.sub(r'if _is_cuda or _is_xpu:', 'if _is_cuda:', content)

    # and not (_is_xpu) -> (remove, always true for CUDA)
    content = re.sub(r'\s+and\s+not\s+\(_is_xpu\)', '', content)

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    srt_dir = Path('/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt')

    files_with_xpu = [
        'layers/layernorm.py',
        'layers/utils/multi_platform.py',
        'layers/quantization/gguf.py',
        'layers/quantization/awq.py',
        'layers/activation.py',
        'layers/rotary_embedding.py',
        'distributed/parallel_state.py',
        'mem_cache/memory_pool_host.py',
    ]

    print("=" * 80)
    print("Removing XPU references (Intel XPU not supported, CUDA-only)")
    print("=" * 80)
    print()

    modified = 0
    for file_rel in files_with_xpu:
        file_path = srt_dir / file_rel
        if file_path.exists():
            if clean_xpu_refs(file_path):
                print(f"✓ {file_rel}")
                modified += 1
            else:
                print(f"  {file_rel}: already clean")
        else:
            print(f"✗ {file_rel}: NOT FOUND")

    print()
    print(f"Modified: {modified}/{len(files_with_xpu)}")
    print()

    # Verify
    count = 0
    for py_file in srt_dir.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                count += f.read().count('_is_xpu')
        except:
            pass

    print(f"Remaining _is_xpu references: {count}")
    if count == 0:
        print("✅ All XPU references removed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
