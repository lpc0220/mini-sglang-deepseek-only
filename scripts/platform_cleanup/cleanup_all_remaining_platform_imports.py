#!/usr/bin/env python3
"""
Comprehensive cleanup of ALL remaining platform-specific imports.
This extends the previous cleanup to cover all files found in validation.
"""

import re
from pathlib import Path

# All files with platform import issues found in validation
files_to_fix = [
    # Original batch (already cleaned, but verify)
    "sglang/python/sglang/srt/server_args.py",
    "sglang/python/sglang/srt/layers/layernorm.py",
    "sglang/python/sglang/srt/layers/vocab_parallel_embedding.py",
    "sglang/python/sglang/srt/layers/communicator.py",
    "sglang/python/sglang/srt/layers/activation.py",
    "sglang/python/sglang/srt/layers/rotary_embedding.py",
    "sglang/python/sglang/srt/distributed/parallel_state.py",

    # New batch from validation
    "sglang/python/sglang/srt/batch_overlap/two_batch_overlap.py",
    "sglang/python/sglang/srt/managers/scheduler_profiler_mixin.py",
    "sglang/python/sglang/srt/managers/mm_utils.py",
    "sglang/python/sglang/srt/model_executor/model_runner.py",
    "sglang/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py",
    "sglang/python/sglang/srt/model_executor/cuda_graph_runner.py",
    "sglang/python/sglang/srt/model_executor/forward_batch_info.py",
    "sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
]

# Platform imports to remove
platform_imports = ["is_npu", "is_xpu", "is_hip", "is_cpu"]

def cleanup_imports(file_path):
    """Remove unused platform imports from a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # Remove platform imports from import lines
    for platform in platform_imports:
        # Pattern 1: Remove from middle of import list (with commas)
        before = content
        content = re.sub(rf',\s*{platform}\s*,', ',', content)
        if content != before:
            changes.append(f"Removed {platform} from middle")
            before = content

        # Pattern 2: Remove from end of import list
        content = re.sub(rf',\s*{platform}\s*\)', ')', content)
        if content != before:
            changes.append(f"Removed {platform} from end")
            before = content

        # Pattern 3: Remove from start of import list
        content = re.sub(rf'\(\s*{platform}\s*,', '(', content)
        if content != before:
            changes.append(f"Removed {platform} from start")
            before = content

        # Pattern 4: Single import on its own line
        content = re.sub(rf'from\s+\S+\s+import\s+{platform}\s*\n', '', content)
        if content != before:
            changes.append(f"Removed standalone {platform} import")
            before = content

    # Clean up double commas
    content = re.sub(r',\s*,', ',', content)

    # Clean up trailing commas before closing paren
    content = re.sub(r',\s*\)', ')', content)

    # Clean up empty parentheses
    content = re.sub(r'import\s+\(\s*\)', 'import ()', content)

    if content != original:
        return content, changes
    return None, []

# Process files
processed = 0
total_changes = 0
for file_rel in files_to_fix:
    file_path = Path(file_rel)
    if file_path.exists():
        result = cleanup_imports(file_path)
        if result[0] is not None:
            new_content, changes = result
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"✅ {file_rel}")
            for change in changes:
                print(f"   - {change}")
            processed += 1
            total_changes += len(changes)
        else:
            print(f"✓  {file_rel}: Already clean")
    else:
        print(f"❌ {file_rel}: File not found")

print(f"\n✅ Processed {processed} files with {total_changes} changes")
