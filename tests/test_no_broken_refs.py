#!/usr/bin/env python3
"""
Test Suite: Broken Reference Detection
Checks for imports of removed models that would cause runtime errors
"""
import os
import re
import sys

SGLANG_ROOT = os.path.join(os.path.dirname(__file__), '..', 'sglang', 'python')

# Models we removed
REMOVED_MODELS = [
    # Batch 1: DeepSeek multimodal
    'deepseek_janus_pro', 'deepseek_ocr', 'deepseek_vl2',

    # Batch 2: Non-DeepSeek models (partial list - main ones)
    'llama', 'qwen', 'mistral', 'mixtral', 'gemma', 'phi',
    'mllama', 'internlm', 'chatglm', 'glm4', 'baichuan',

    # Batch 3: Model stragglers
    'llama_eagle', 'llama_embedding', 'llama_reward',
    'longcat_flash', 'minicpm', 'nano_nemotron_vl',
]

def find_imports_of_removed_models():
    """Scan codebase for imports of removed models"""
    broken_refs = []

    for root, dirs, files in os.walk(SGLANG_ROOT):
        # Skip test directories and __pycache__
        if '__pycache__' in root or '/test/' in root:
            continue

        for file in files:
            if not file.endswith('.py'):
                continue

            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                for removed in REMOVED_MODELS:
                    # Look for imports like: from sglang.srt.models.llama import
                    # or: from sglang.srt.models import llama
                    patterns = [
                        rf'from\s+sglang\.srt\.models\.{removed}\s+import',
                        rf'from\s+sglang\.srt\.models\s+import.*{removed}',
                        rf'import\s+sglang\.srt\.models\.{removed}',
                    ]

                    for pattern in patterns:
                        if re.search(pattern, content):
                            broken_refs.append({
                                'file': filepath.replace(SGLANG_ROOT + '/', ''),
                                'model': removed,
                                'pattern': pattern
                            })
            except Exception as e:
                print(f"⚠️  Warning: Could not read {filepath}: {e}")

    return broken_refs

def main():
    print("=" * 60)
    print("Broken Reference Detection")
    print("=" * 60)

    print("\n🔍 Scanning for imports of removed models...")
    broken_refs = find_imports_of_removed_models()

    if not broken_refs:
        print("✅ PASS: No broken references found!")
        return True
    else:
        print(f"\n❌ FAIL: Found {len(broken_refs)} broken reference(s):\n")
        for ref in broken_refs:
            print(f"  File: {ref['file']}")
            print(f"  Model: {ref['model']}")
            print(f"  Pattern: {ref['pattern']}")
            print()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
