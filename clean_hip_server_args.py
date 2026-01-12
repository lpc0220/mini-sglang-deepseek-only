#!/usr/bin/env python3
"""Clean HIP/AMD/ROCm references from server_args.py"""

import re
from pathlib import Path

def clean_server_args():
    filepath = Path("/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt/server_args.py")
    print(f"Processing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()
        original_lines = content.count('\n')

    original_content = content

    # Remove is_hip import
    content = re.sub(r',\s*is_hip(?=\s*,)', '', content)
    content = re.sub(r'is_hip,\s*', '', content)

    # Remove elif is_hip() blocks in attention backend selection
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

        # Check for elif is_hip():
        if re.search(r'elif is_hip\(\):', line):
            indent = len(line) - len(line.lstrip())
            skip_until_indent = indent
            i += 1
            continue

        # Check for if is_hip():
        if re.search(r'^\s+if is_hip\(\):', line):
            indent = len(line) - len(line.lstrip())
            skip_until_indent = indent
            i += 1
            continue

        cleaned_lines.append(line)
        i += 1

    content = '\n'.join(cleaned_lines)

    # Remove _handle_amd_specifics() call
    content = re.sub(r'\s+self\._handle_amd_specifics\(\)\n', '', content)

    # Remove _handle_amd_specifics method definition (complex multi-line)
    # Find the method and remove it
    content = re.sub(
        r'    def _handle_amd_specifics\(self\):.*?(?=\n    def |\nclass |\Z)',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove AMD-specific comments
    content = re.sub(r'\s*# AMD.*?\n', '\n', content)
    content = re.sub(r'\s*# On AMD/HIP.*?\n', '\n', content)
    content = re.sub(r".*'aiter'.*ROCm only.*?\n", '', content)

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

if __name__ == "__main__":
    changed, lines_removed = clean_server_args()
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files changed: {1 if changed else 0}/1")
    print(f"  Total lines removed: {lines_removed}")
    print(f"{'='*60}")
