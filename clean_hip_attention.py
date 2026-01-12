#!/usr/bin/env python3
"""Clean HIP/AMD/ROCm references from attention layer files."""

import re
from pathlib import Path

def clean_file(filepath: Path) -> tuple[bool, int]:
    """Clean HIP references from a file. Returns (changed, lines_removed)."""
    print(f"Processing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()
        original_lines = content.count('\n')

    original_content = content

    # Pattern 1: Remove `_is_hip = is_hip()` module variable
    content = re.sub(r'^_is_hip = is_hip\(\)\n', '', content, flags=re.MULTILINE)

    # Pattern 2: Remove `is_hip` from imports
    content = re.sub(r',\s*is_hip(?=\s*[,)])', '', content)
    content = re.sub(r'is_hip,\s*', '', content)

    # Pattern 3: Simplify `if _is_cuda or _is_hip:` to `if _is_cuda:`
    content = re.sub(r'if _is_cuda or _is_hip:', 'if _is_cuda:', content)

    # Pattern 4: Simplify `elif _is_hip or _is_cuda:` to `elif _is_cuda:`
    content = re.sub(r'elif _is_hip or _is_cuda:', 'elif _is_cuda:', content)

    # Pattern 5: Remove `elif _is_hip:` blocks
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_indent = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # If we're skipping HIP block
        if skip_until_indent is not None:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= skip_until_indent:
                skip_until_indent = None
            else:
                i += 1
                continue

        # Check for elif _is_hip:
        if re.search(r'elif _is_hip:', line):
            indent = len(line) - len(line.lstrip())
            skip_until_indent = indent
            i += 1
            continue

        # Check for standalone `if _is_hip:`
        if re.search(r'^\s+if _is_hip:', line):
            indent = len(line) - len(line.lstrip())
            skip_until_indent = indent
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    content = '\n'.join(cleaned_lines)

    # Pattern 6: Remove `and not _is_hip`
    content = re.sub(r' and not _is_hip', '', content)

    # Pattern 7: Remove `or _is_hip` in conditions
    content = re.sub(r' or _is_hip', '', content)

    # Write if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        new_lines = content.count('\n')
        lines_removed = original_lines - new_lines
        print(f"  ✓ Cleaned: {lines_removed} lines removed")
        return True, lines_removed
    else:
        print(f"  • No changes needed")
        return False, 0

def main():
    base_path = Path("/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt/layers/attention")

    files_to_clean = [
        "triton_ops/prefill_attention.py",
        "triton_ops/decode_attention.py",
        "triton_ops/extend_attention.py",
        "triton_ops/double_sparsity_attention.py",
        "nsa/tilelang_kernel.py",
        "nsa/nsa_indexer.py",
        "nsa_backend.py",
    ]

    total_changed = 0
    total_lines_removed = 0

    for file_rel_path in files_to_clean:
        filepath = base_path / file_rel_path
        if filepath.exists():
            changed, lines_removed = clean_file(filepath)
            if changed:
                total_changed += 1
                total_lines_removed += lines_removed
        else:
            print(f"WARNING: File not found: {filepath}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files changed: {total_changed}/{len(files_to_clean)}")
    print(f"  Total lines removed: {total_lines_removed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
