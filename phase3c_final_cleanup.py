#!/usr/bin/env python3
"""
Phase 3C Final: Manual cleanup of remaining NPU/HIP references

Handles edge cases like:
- if not _is_npu:
- @torch.compile(disable=_is_npu)
- device = "cuda" if not _is_npu else "npu"
"""

import re
from pathlib import Path

def clean_file(file_path: Path) -> tuple[bool, int]:
    """Clean remaining NPU/HIP references from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = 0

    # Pattern 1: if not _is_npu: -> no condition (always true for CUDA)
    # Remove the entire condition
    pattern1 = re.compile(r'(\s*)if not _is_npu:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    content = pattern1.sub(r'\1\2', content)
    if content != original_content:
        changes += len(pattern1.findall(original_content))
        original_content = content

    # Pattern 2: @torch.compile(disable=_is_npu) -> @torch.compile(disable=False)
    pattern2 = re.compile(r'@torch\.compile\((.*?)disable=_is_npu(.*?)\)')
    content = pattern2.sub(r'@torch.compile(\1disable=False\2)', content)
    if content != original_content:
        changes += len(pattern2.findall(original_content))
        original_content = content

    # Pattern 3: device = "cuda" if not _is_npu else "npu" -> device = "cuda"
    pattern3 = re.compile(r'device:\s*Optional\[str\]\s*=\s*"cuda"\s*if\s*not\s*_is_npu\s*else\s*"npu"')
    content = pattern3.sub(r'device: Optional[str] = "cuda"', content)
    if content != original_content:
        changes += len(pattern3.findall(original_content))
        original_content = content

    # Pattern 4: if _is_npu and ... -> if False and ... (will be optimized away)
    pattern4 = re.compile(r'if _is_npu and ')
    content = pattern4.sub('if False and ', content)
    if content != original_content:
        changes += len(pattern4.findall(original_content))
        original_content = content

    # Pattern 5: or _is_npu in conditions
    pattern5 = re.compile(r'\s+or\s+_is_npu(?=[:\)])')
    content = pattern5.sub('', content)
    if content != original_content:
        changes += len(pattern5.findall(original_content))
        original_content = content

    # Pattern 6: not (_is_cuda or _is_npu) -> not _is_cuda
    pattern6 = re.compile(r'not\s+\(_is_cuda\s+or\s+_is_npu\)')
    content = pattern6.sub('not _is_cuda', content)
    if content != original_content:
        changes += len(pattern6.findall(original_content))
        original_content = content

    # Pattern 7: return [...] if not _is_npu else [...] -> return [...]
    pattern7 = re.compile(r'return\s+(\[.*?\])\s+if\s+not\s+_is_npu\s+else\s+\[.*?\]')
    content = pattern7.sub(r'return \1', content)
    if content != original_content:
        changes += len(pattern7.findall(original_content))
        original_content = content

    # Pattern 8: if self... and _is_npu: -> if False:
    pattern8 = re.compile(r'if\s+(.*?)\s+and\s+_is_npu:')
    content = pattern8.sub(r'if False:  # NPU removed', content)
    if content != original_content:
        changes += len(pattern8.findall(original_content))
        original_content = content

    # Pattern 9: if _is_npu and (...): -> if False:
    pattern9 = re.compile(r'if\s+_is_npu\s+and\s+\(')
    content = pattern9.sub('if False and (', content)
    if content != original_content:
        changes += len(pattern9.findall(original_content))
        original_content = content

    # HIP patterns
    # Pattern 10: if not _is_hip:
    pattern10 = re.compile(r'(\s*)if not _is_hip:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    content = pattern10.sub(r'\1\2', content)
    if content != original_content:
        changes += len(pattern10.findall(original_content))
        original_content = content

    # Pattern 11: @torch.compile(disable=_is_hip)
    pattern11 = re.compile(r'@torch\.compile\((.*?)disable=_is_hip(.*?)\)')
    content = pattern11.sub(r'@torch.compile(\1disable=False\2)', content)
    if content != original_content:
        changes += len(pattern11.findall(original_content))
        original_content = content

    # Pattern 12: if _is_hip and
    pattern12 = re.compile(r'if _is_hip and ')
    content = pattern12.sub('if False and ', content)
    if content != original_content:
        changes += len(pattern12.findall(original_content))
        original_content = content

    # Pattern 13: or _is_hip
    pattern13 = re.compile(r'\s+or\s+_is_hip(?=[:\)])')
    content = pattern13.sub('', content)
    if content != original_content:
        changes += len(pattern13.findall(original_content))
        original_content = content

    # Pattern 14: if not _use_aiter:
    pattern14 = re.compile(r'(\s*)if not _use_aiter:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)')
    content = pattern14.sub(r'\1\2', content)
    if content != original_content:
        changes += len(pattern14.findall(original_content))
        original_content = content

    if content != open(file_path, 'r').read():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes

    return False, 0


def main():
    srt_dir = Path('/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt')

    # Files with remaining references
    files_to_clean = [
        'layers/logits_processor.py',
        'layers/vocab_parallel_embedding.py',
        'layers/quantization/awq.py',
        'layers/communicator.py',
        'layers/rotary_embedding.py',
        'layers/moe/token_dispatcher/deepep.py',
        'layers/moe/moe_runner/deep_gemm.py',
        'layers/moe/topk.py',
        'managers/scheduler_profiler_mixin.py',
        'model_executor/model_runner.py',
        'model_executor/forward_batch_info.py',
        'model_executor/model_runner_kv_cache_mixin.py',
        'utils/profile_utils.py',
    ]

    print("=" * 80)
    print("Phase 3C Final Cleanup: Remaining NPU/HIP References")
    print("=" * 80)
    print()

    total_modified = 0
    total_changes = 0

    for file_rel in files_to_clean:
        file_path = srt_dir / file_rel
        if file_path.exists():
            modified, changes = clean_file(file_path)
            if modified:
                print(f"✓ {file_rel}: {changes} patterns fixed")
                total_modified += 1
                total_changes += changes
            else:
                print(f"  {file_rel}: already clean")
        else:
            print(f"✗ {file_rel}: NOT FOUND")

    print()
    print("=" * 80)
    print(f"Files modified: {total_modified}/{len(files_to_clean)}")
    print(f"Patterns fixed: {total_changes}")
    print("=" * 80)


if __name__ == '__main__':
    main()
