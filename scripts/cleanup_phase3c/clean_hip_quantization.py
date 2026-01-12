#!/usr/bin/env python3
"""
Clean HIP/AMD/ROCm references from quantization files.
CRITICAL: Remove ONLY conditional branches, NOT implementations!
"""

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

    # Pattern 2: Remove `_use_aiter = get_bool_env_var(...) and _is_hip`
    content = re.sub(r'^_use_aiter = get_bool_env_var\([^)]+\) and _is_hip\n', '', content, flags=re.MULTILINE)

    # Pattern 3: Remove `is_hip` from imports
    content = re.sub(r',\s*is_hip(?=\s*[,)])', '', content)
    content = re.sub(r'is_hip,\s*', '', content)

    # Pattern 4: Simplify `if _is_cuda or _is_hip:` to `if _is_cuda:`
    content = re.sub(r'if _is_cuda or _is_hip:', 'if _is_cuda:', content)

    # Pattern 5: Remove `elif _is_hip:` blocks (keep the else block if present)
    # This is more complex - need to handle indentation
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_indent = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # If we're skipping HIP block
        if skip_until_indent is not None:
            current_indent = len(line) - len(line.lstrip())
            # Skip until we reach same or lesser indentation
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

        # Check for `if _is_hip:` standalone (not elif)
        if re.search(r'^\s+if _is_hip:', line):
            indent = len(line) - len(line.lstrip())
            # Look ahead for else block
            j = i + 1
            else_found = False
            while j < len(lines):
                next_line = lines[j]
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_line.strip().startswith('else:') and next_indent == indent:
                    else_found = True
                    # Skip to else block
                    skip_until_indent = indent
                    break
                elif next_line.strip() and next_indent <= indent:
                    # End of if block, no else
                    skip_until_indent = indent
                    break
                j += 1
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    content = '\n'.join(cleaned_lines)

    # Pattern 6: Remove aiter imports
    content = re.sub(r'if _use_aiter:.*?except ImportError:.*?pass\n', '', content, flags=re.DOTALL)
    content = re.sub(r'from aiter import.*?\n', '', content)

    # Pattern 7: Remove _use_aiter references
    content = re.sub(r' and _use_aiter', '', content)
    content = re.sub(r'_use_aiter and ', '', content)
    content = re.sub(r'if _use_aiter or ', 'if ', content)
    content = re.sub(r' or _use_aiter', '', content)

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
    base_path = Path("/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt/layers/quantization")

    files_to_clean = [
        "gguf.py",
        "fp8.py",
        "unquant.py",
        "fp8_utils.py",
        "fp8_kernel.py",
        "__init__.py",
        "awq.py",
        "mxfp4.py",
        "quark/schemes/quark_w4a4_mxfp4.py",
        "quark/schemes/quark_w8a8_fp8.py",
        "quark/quark_moe.py",
        "petit.py",
        "compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py",
        "compressed_tensors/compressed_tensors_moe.py",
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
