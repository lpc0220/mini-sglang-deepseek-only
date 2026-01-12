#!/bin/bash
# Run all validation tests

set -e

echo "========================================"
echo "SGLang DeepSeek-Only Validation Suite"
echo "========================================"
echo ""

cd "$(dirname "$0")"

echo "Test 1: Import Validation"
echo "----------------------------------------"
python3 test_imports.py
echo ""

echo "Test 2: Broken Reference Detection"
echo "----------------------------------------"
python3 test_no_broken_refs.py
echo ""

echo "========================================"
echo "✅ All validation tests passed!"
echo "========================================"
