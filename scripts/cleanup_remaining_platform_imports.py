#!/usr/bin/env python3
"""
Cleanup remaining platform-specific imports.
These are imports that are no longer used after platform code removal.
"""

import re
from pathlib import Path

# Files to clean up
files_to_fix = [
    "sglang/python/sglang/srt/server_args.py",
    "sglang/python/sglang/srt/layers/layernorm.py",
    "sglang/python/sglang/srt/layers/vocab_parallel_embedding.py",
    "sglang/python/sglang/srt/layers/communicator.py",
    "sglang/python/sglang/srt/layers/activation.py",
    "sglang/python/sglang/srt/layers/rotary_embedding.py",
    "sglang/python/sglang/srt/distributed/parallel_state.py",
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
        content = re.sub(rf',\s*{platform}\s*,', ',', content)

        # Pattern 2: Remove from end of import list
        content = re.sub(rf',\s*{platform}\s*\)', ')', content)

        # Pattern 3: Remove from start of import list
        content = re.sub(rf'\(\s*{platform}\s*,', '(', content)

        if content != original:
            changes.append(f"Removed {platform}")
            original = content

    # Clean up double commas
    content = re.sub(r',\s*,', ',', content)

    # Clean up trailing commas before closing paren
    content = re.sub(r',\s*\)', ')', content)

    return content, changes

# Process files
processed = 0
for file_rel in files_to_fix:
    file_path = Path(file_rel)
    if file_path.exists():
        new_content, changes = cleanup_imports(file_path)
        if changes:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"✅ {file_rel}: {', '.join(changes)}")
            processed += 1
        else:
            print(f"⚠️  {file_rel}: No changes needed")
    else:
        print(f"❌ {file_rel}: File not found")

print(f"\n✅ Processed {processed} files")
