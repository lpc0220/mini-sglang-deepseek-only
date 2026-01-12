#!/usr/bin/env python3
"""
Test Suite: Syntax and Basic Import Validation
Validates Python syntax and basic import structure without dependencies
"""
import os
import py_compile
import sys
import ast

SGLANG_ROOT = os.path.join(os.path.dirname(__file__), '..', 'sglang', 'python')

def test_syntax_errors():
    """Check for Python syntax errors in key files"""
    key_files = [
        'sglang/srt/models/deepseek.py',
        'sglang/srt/models/deepseek_v2.py',
        'sglang/srt/models/deepseek_nextn.py',
        'sglang/srt/models/registry.py',
        'sglang/srt/models/utils.py',
    ]

    errors = []
    for file in key_files:
        filepath = os.path.join(SGLANG_ROOT, file)
        if not os.path.exists(filepath):
            errors.append(f"Missing file: {file}")
            continue

        try:
            py_compile.compile(filepath, doraise=True)
            print(f"✅ {file} - syntax OK")
        except py_compile.PyCompileError as e:
            errors.append(f"{file}: {e}")
            print(f"❌ {file} - syntax error: {e}")

    return len(errors) == 0, errors

def test_no_removed_model_imports():
    """Check that DeepSeek models don't import removed models"""
    removed_models = [
        'deepseek_janus_pro', 'deepseek_ocr', 'deepseek_vl2',
        'llama', 'qwen', 'mistral', 'gemma',
    ]

    deepseek_files = [
        'sglang/srt/models/deepseek.py',
        'sglang/srt/models/deepseek_v2.py',
        'sglang/srt/models/deepseek_nextn.py',
    ]

    issues = []
    for file in deepseek_files:
        filepath = os.path.join(SGLANG_ROOT, file)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        for removed in removed_models:
            if f'from .{removed} import' in content or f'import .{removed}' in content:
                issues.append(f"{file} imports removed model: {removed}")

    if not issues:
        print("✅ No imports of removed models in DeepSeek files")
    else:
        for issue in issues:
            print(f"❌ {issue}")

    return len(issues) == 0, issues

def test_deepseek_common_exists():
    """Check that deepseek_common directory exists"""
    common_dir = os.path.join(SGLANG_ROOT, 'sglang/srt/models/deepseek_common')
    if os.path.isdir(common_dir):
        print(f"✅ deepseek_common/ directory exists")
        # List files in it
        files = [f for f in os.listdir(common_dir) if f.endswith('.py')]
        print(f"   Found {len(files)} Python files")
        return True, []
    else:
        print(f"❌ deepseek_common/ directory missing!")
        return False, ["Missing deepseek_common directory"]

def main():
    print("=" * 60)
    print("Syntax and Structure Validation")
    print("=" * 60)
    print()

    all_passed = True
    all_errors = []

    print("Test 1: Python Syntax Check")
    print("-" * 60)
    passed, errors = test_syntax_errors()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("Test 2: No Removed Model Imports")
    print("-" * 60)
    passed, errors = test_no_removed_model_imports()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("Test 3: DeepSeek Common Directory")
    print("-" * 60)
    passed, errors = test_deepseek_common_exists()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    print("=" * 60)
    if all_passed:
        print("✅ All validation tests PASSED")
    else:
        print(f"❌ Tests FAILED with {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  - {err}")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
