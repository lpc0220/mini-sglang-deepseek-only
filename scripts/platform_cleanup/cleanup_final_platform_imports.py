#!/usr/bin/env python3
"""
Final cleanup of platform-specific imports.
"""

import re
from pathlib import Path

# Final batch from validation
files_to_fix = [
    "sglang/python/sglang/srt/utils/profile_utils.py",
    "sglang/python/sglang/srt/models/deepseek_nextn.py",
    "sglang/python/sglang/srt/constrained/xgrammar_backend.py",
    "sglang/python/sglang/srt/mem_cache/memory_pool.py",
    "sglang/python/sglang/srt/mem_cache/memory_pool_host.py",
    "sglang/python/sglang/srt/configs/model_config.py",
    "sglang/python/sglang/srt/configs/load_config.py",
    "sglang/python/sglang/srt/model_loader/loader.py",
    "sglang/python/sglang/srt/speculative/eagle_info_v2.py",
]

platform_imports = ["is_npu", "is_xpu", "is_hip", "is_cpu"]

def cleanup_imports(file_path):
    """Remove unused platform imports from a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original = content
    changes = []

    for platform in platform_imports:
        before = content

        # Remove from comma-separated imports
        content = re.sub(rf',\s*{platform}\s*,', ', ', content)
        content = re.sub(rf',\s*{platform}(?=\s*\))', '', content)
        content = re.sub(rf'(?<=\()\s*{platform}\s*,\s*', '', content)
        content = re.sub(rf',\s*{platform}(?=\s*$)', '', content, flags=re.MULTILINE)

        # Remove standalone import
        content = re.sub(rf'from\s+[\w.]+\s+import\s+{platform}\s*\n', '', content)

        if content != before:
            changes.append(f"Removed {platform}")

    # Clean up double commas and spaces
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r',\s*\)', ')', content)

    if content != original:
        return content, changes
    return None, []

processed = 0
for file_rel in files_to_fix:
    file_path = Path(file_rel)
    if file_path.exists():
        result = cleanup_imports(file_path)
        if result[0] is not None:
            new_content, changes = result
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"✅ {file_path.name}: {', '.join(changes)}")
            processed += 1
        else:
            print(f"✓  {file_path.name}: Already clean")
    else:
        print(f"❌ {file_rel}: File not found")

print(f"\n✅ Processed {processed} files")
