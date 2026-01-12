#!/usr/bin/env python3
"""
Comprehensive cleanup of ALL platform-specific imports in the entire codebase.
"""

import re
from pathlib import Path

# All remaining files with platform imports
files_to_fix = [
    "sglang/python/sglang/srt/layers/quantization/awq.py",
    "sglang/python/sglang/srt/layers/moe/ep_moe/layer.py",
    "sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py",
    "sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py",
    "sglang/python/sglang/srt/distributed/device_communicators/torch_symm_mem.py",
    "sglang/python/sglang/srt/distributed/device_communicators/pymscclpp.py",
    "sglang/python/sglang/srt/distributed/device_communicators/quick_all_reduce.py",
    "sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py",
    "sglang/python/sglang/srt/multimodal/processors/base_processor.py",
    "sglang/python/sglang/srt/speculative/spec_utils.py",
    "sglang/python/sglang/srt/speculative/eagle_utils.py",
    "sglang/python/sglang/srt/speculative/ngram_info.py",
    "sglang/python/sglang/srt/speculative/multi_layer_eagle_worker.py",
    "sglang/python/sglang/srt/compilation/backend.py",
]

platform_imports = ["is_npu", "is_xpu", "is_hip", "is_cpu"]

def cleanup_imports(file_path):
    """Remove unused platform imports from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    original_content = ''.join(lines)
    changes = []

    # Process each line
    new_lines = []
    for line in lines:
        original_line = line
        modified = False

        # Check if this is an import line with platform imports
        if 'import' in line and any(platform in line for platform in platform_imports):
            for platform in platform_imports:
                if platform in line:
                    # Remove the platform import
                    # Handle various patterns
                    line = re.sub(rf',\s*{platform}\b', '', line)  # Middle or end
                    line = re.sub(rf'\b{platform}\s*,\s*', '', line)  # Beginning
                    line = re.sub(rf'\bimport\s+{platform}\s*$', '', line)  # Standalone
                    if line != original_line:
                        modified = True
                        changes.append(f"Removed {platform}")

            # Clean up any double commas or trailing commas
            line = re.sub(r',\s*,', ',', line)
            line = re.sub(r',\s*\)', ')', line)
            line = re.sub(r'\(\s*\)', '()', line)

            # If the import line is now empty (only "from X import"), skip it
            if re.match(r'^\s*from\s+[\w.]+\s+import\s*\(\s*\)\s*$', line):
                continue

        new_lines.append(line)

    new_content = ''.join(new_lines)

    if new_content != original_content:
        return new_content, changes
    return None, []

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
            print(f"✅ {file_path.name}: {', '.join(set(changes))}")
            processed += 1
            total_changes += len(changes)
        else:
            print(f"✓  {file_path.name}: Already clean")
    else:
        print(f"❌ {file_rel}: File not found")

print(f"\n✅ Processed {processed} files with {total_changes} changes")
