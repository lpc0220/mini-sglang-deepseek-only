#!/usr/bin/env python3
"""
Phase 3C Ultra Final: Clean ALL remaining NPU/HIP references

This handles all edge cases including:
- if _is_hip and Lk >= 576:
- max_warps = 16 if _is_hip else 32
- if not _is_hip:
- _use_hip_int4 = ... and _is_hip
- IS_CUSTOM_AR_AVAILABLE = _is_cuda or _is_hip
- dynamic=_is_hip and ...
- if not (_is_npu or _is_xpu):
"""

import re
from pathlib import Path
from typing import List, Tuple

def clean_file_comprehensive(file_path: Path) -> Tuple[bool, int, List[str]]:
    """Comprehensively clean all NPU/HIP references."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # HIP Patterns

    # 1. if _is_hip and condition: -> if False and condition:
    pattern = re.compile(r'if _is_hip and ')
    if pattern.search(content):
        content = pattern.sub('if False and ', content)
        changes.append('if _is_hip and -> if False and')

    # 2. max_warps = 16 if _is_hip else 32 -> max_warps = 32
    pattern = re.compile(r'(\w+)\s*=\s*\d+\s*if\s*_is_hip\s*else\s*(\d+)')
    if pattern.search(content):
        content = pattern.sub(r'\1 = \2', content)
        changes.append('X if _is_hip else Y -> X = Y')

    # 3. if not _is_hip: (remove condition)
    pattern = re.compile(r'(\s*)if not _is_hip:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    if pattern.search(content):
        content = pattern.sub(r'\1\2', content)
        changes.append('if not _is_hip: (removed)')

    # 4. _use_hip_int4 = ... and _is_hip -> _use_hip_int4 = False
    pattern = re.compile(r'_use_hip_int4\s*=\s*.*?and\s*_is_hip')
    if pattern.search(content):
        content = pattern.sub('_use_hip_int4 = False', content)
        changes.append('_use_hip_int4 = ... and _is_hip -> False')

    # 5. if _is_hip and _use_hip_int4: -> if False:
    pattern = re.compile(r'if _is_hip and _use_hip_int4:')
    if pattern.search(content):
        content = pattern.sub('if False:', content)
        changes.append('if _is_hip and _use_hip_int4: -> if False:')

    # 6. if _is_hip and condition: -> if False:
    pattern = re.compile(r'if _is_hip and .*?:')
    if pattern.search(content):
        content = pattern.sub('if False:', content)
        changes.append('if _is_hip and ...: -> if False:')

    # 7. IS_X = _is_cuda or _is_hip -> IS_X = _is_cuda
    pattern = re.compile(r'(IS_\w+\s*=\s*)_is_cuda\s+or\s+_is_hip')
    if pattern.search(content):
        content = pattern.sub(r'\1_is_cuda', content)
        changes.append('IS_X = _is_cuda or _is_hip -> _is_cuda')

    # 8. IS_X = _is_hip -> IS_X = False
    pattern = re.compile(r'(IS_\w+\s*=\s*)_is_hip')
    if pattern.search(content):
        content = pattern.sub(r'\1False', content)
        changes.append('IS_X = _is_hip -> False')

    # 9. _USE_X = _is_hip and ... -> _USE_X = False
    pattern = re.compile(r'(_USE_\w+\s*=\s*)_is_hip\s+and\s+.*')
    if pattern.search(content):
        content = pattern.sub(r'\1False', content)
        changes.append('_USE_X = _is_hip and ... -> False')

    # 10. if _is_hip: (without else) -> if False:
    pattern = re.compile(r'if _is_hip:')
    if pattern.search(content):
        content = pattern.sub('if False:', content)
        changes.append('if _is_hip: -> if False:')

    # 11. dynamic=_is_hip and ... -> dynamic=False
    pattern = re.compile(r'dynamic=_is_hip\s+and\s+[^,)]+')
    if pattern.search(content):
        content = pattern.sub('dynamic=False', content)
        changes.append('dynamic=_is_hip and ... -> False')

    # NPU Patterns

    # 12. if not _is_npu: (remove condition)
    pattern = re.compile(r'(\s*)if not _is_npu:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    if pattern.search(content):
        content = pattern.sub(r'\1\2', content)
        changes.append('if not _is_npu: (removed)')

    # 13. elif not _is_npu: -> (remove, keep code)
    pattern = re.compile(r'(\s*)elif not _is_npu:\n((?:(?!\1(?:elif|else|def|class|@)).*\n)*)')
    if pattern.search(content):
        content = pattern.sub(r'\1\2', content)
        changes.append('elif not _is_npu: (removed)')

    # 14. or _is_npu in conditions
    pattern = re.compile(r'\s+or\s+_is_npu')
    if pattern.search(content):
        content = pattern.sub('', content)
        changes.append('or _is_npu (removed)')

    # 15. and not _is_npu -> (remove, always true)
    pattern = re.compile(r'\s+and\s+not\s+_is_npu')
    if pattern.search(content):
        content = pattern.sub('', content)
        changes.append('and not _is_npu (removed)')

    # 16. if (_condition) or _is_npu:
    pattern = re.compile(r'if\s+\((.*?)\)\s+or\s+_is_npu:')
    if pattern.search(content):
        content = pattern.sub(r'if \1:', content)
        changes.append('if (cond) or _is_npu: -> if cond:')

    # 17. @torch.compile(disable=_is_npu) -> @torch.compile(disable=False)
    pattern = re.compile(r'@torch\.compile\((.*?)disable=_is_npu(.*?)\)')
    if pattern.search(content):
        content = pattern.sub(r'@torch.compile(\1disable=False\2)', content)
        changes.append('@torch.compile(disable=_is_npu) -> False')

    # 18. if not (_is_npu or _is_xpu): -> (remove condition, always true for CUDA)
    pattern = re.compile(r'(\s*)if not \(_is_npu or _is_xpu\):\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    if pattern.search(content):
        content = pattern.sub(r'\1\2', content)
        changes.append('if not (_is_npu or _is_xpu): (removed)')

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, len(changes), changes

    return False, 0, []


def main():
    srt_dir = Path('/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt')

    print("=" * 80)
    print("Phase 3C Ultra Final Cleanup: ALL Remaining NPU/HIP References")
    print("=" * 80)
    print()

    # Find all files with remaining refs
    files_with_refs = []
    for py_file in srt_dir.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if '_is_npu' in content or '_is_hip' in content:
                    files_with_refs.append(py_file)
        except:
            pass

    print(f"Found {len(files_with_refs)} files with NPU/HIP references")
    print()

    total_modified = 0
    total_changes = 0

    for file_path in sorted(files_with_refs):
        rel_path = file_path.relative_to(srt_dir)
        modified, num_changes, change_list = clean_file_comprehensive(file_path)
        if modified:
            print(f"✓ {rel_path}")
            for change in change_list:
                print(f"  - {change}")
            total_modified += 1
            total_changes += num_changes
        else:
            print(f"  {rel_path}: no patterns matched")

    print()
    print("=" * 80)
    print(f"Files modified: {total_modified}/{len(files_with_refs)}")
    print(f"Total changes: {total_changes}")
    print("=" * 80)

    # Final verification
    print()
    print("Final verification...")
    npu_count = 0
    hip_count = 0
    for py_file in srt_dir.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                npu_count += content.count('_is_npu')
                hip_count += content.count('_is_hip')
        except:
            pass

    print(f"Remaining _is_npu: {npu_count}")
    print(f"Remaining _is_hip: {hip_count}")
    print()

    if npu_count == 0 and hip_count == 0:
        print("✅ SUCCESS: All NPU and HIP conditionals removed!")
    else:
        print(f"⚠️  {npu_count + hip_count} references remain (checking if in imports/comments)")
        # Show remaining context
        import subprocess
        result = subprocess.run(
            ['grep', '-rn', '_is_npu\\|_is_hip', str(srt_dir), '--include=*.py'],
            capture_output=True, text=True
        )
        lines = result.stdout.split('\n')[:10]
        print("\nFirst 10 remaining references:")
        for line in lines:
            if line:
                print(f"  {line}")


if __name__ == '__main__':
    main()
