#!/bin/bash
# DeepSeek-R1-NVFP4-v2 Kernel Benchmarks
#
# Usage:
#   ./run.sh              # Run all 23 kernels
#   ./run.sh --list       # List available kernels
#   ./run.sh rmsnorm      # Run specific kernel(s)
#
# Results saved to: results/

cd "$(dirname "$0")/scripts"

if [ "$1" == "--list" ]; then
    python run_all_benchmarks.py --list
elif [ -n "$1" ]; then
    # Run specific kernels (comma-separated)
    python run_all_benchmarks.py --kernels "$1" --output ../results/
else
    # Run all benchmarks
    python run_all_benchmarks.py --output ../results/
fi
