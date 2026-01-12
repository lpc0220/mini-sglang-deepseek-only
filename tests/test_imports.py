#!/usr/bin/env python3
"""
Test Suite: Import Validation
Tests that DeepSeek models and dependencies can be imported correctly
"""
import sys
import os

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sglang', 'python'))

def test_deepseek_v2_import():
    """Test that DeepSeek v2 model can be imported"""
    try:
        from sglang.srt.models import deepseek_v2
        print("✅ PASS: deepseek_v2 imports successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: deepseek_v2 import failed: {e}")
        return False

def test_deepseek_import():
    """Test that DeepSeek base model can be imported"""
    try:
        from sglang.srt.models import deepseek
        print("✅ PASS: deepseek imports successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: deepseek import failed: {e}")
        return False

def test_deepseek_nextn_import():
    """Test that DeepSeek nextn model can be imported"""
    try:
        from sglang.srt.models import deepseek_nextn
        print("✅ PASS: deepseek_nextn imports successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: deepseek_nextn import failed: {e}")
        return False

def test_model_registry_import():
    """Test that model registry can be imported"""
    try:
        from sglang.srt.models import registry
        print("✅ PASS: model registry imports successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: model registry import failed: {e}")
        return False

def test_function_call_import():
    """Test that function calling infrastructure can be imported"""
    try:
        from sglang.srt import function_call
        print("✅ PASS: function_call module imports successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: function_call import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Import Validation Tests")
    print("=" * 60)

    tests = [
        test_deepseek_v2_import,
        test_deepseek_import,
        test_deepseek_nextn_import,
        test_model_registry_import,
        test_function_call_import,
    ]

    results = []
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        results.append(test())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
