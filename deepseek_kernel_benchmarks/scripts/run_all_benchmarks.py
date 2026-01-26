#!/usr/bin/env python3
"""
Run all DeepSeek-R1-NVFP4-v2 kernel benchmarks.

This script orchestrates running all 23 kernel benchmarks and aggregates results.

Usage:
    cd deepseek_kernel_benchmarks/scripts
    python run_all_benchmarks.py --output ../results/
    python run_all_benchmarks.py --kernels rmsnorm,cutlass_scaled_fp4_mm --output ../results/
"""

import argparse
import csv
import importlib
import os
import sys
from datetime import datetime
from typing import List

# All 23 kernels in execution order
KERNELS = [
    # Normalization (2)
    ("bench_rmsnorm", "rmsnorm", "Norm", "Memory"),
    ("bench_fused_add_rmsnorm", "fused_add_rmsnorm", "Norm", "Memory"),

    # GEMM (3)
    ("bench_cutlass_scaled_fp4_mm", "cutlass_scaled_fp4_mm", "GEMM", "Compute"),
    ("bench_dsv3_fused_a_gemm", "dsv3_fused_a_gemm", "GEMM", "Compute"),
    ("bench_dsv3_router_gemm", "dsv3_router_gemm", "GEMM", "Compute"),

    # BMM (1)
    ("bench_bmm_fp8", "bmm_fp8", "BMM", "Compute"),

    # Attention (4)
    ("bench_cutlass_mla_decode", "cutlass_mla_decode", "Attention", "Mixed"),
    ("bench_trtllm_mla", "trtllm_batch_decode_with_kv_cache_mla", "Attention", "Mixed"),
    ("bench_trtllm_ragged_attention", "trtllm_ragged_attention_deepseek", "Attention", "Mixed"),
    ("bench_mla_rope_quantize_fp8", "mla_rope_quantize_fp8", "Attention", "Memory"),

    # RoPE & Concat (2)
    ("bench_apply_rope", "apply_rope_with_cos_sin_cache_inplace", "RoPE", "Memory"),
    ("bench_concat_mla_mha_k", "concat_mla_mha_k", "Concat", "Memory"),

    # Activation (1)
    ("bench_silu_and_mul", "silu_and_mul", "Activation", "Memory"),

    # MoE Routing (3)
    ("bench_topk_softmax", "topk_softmax", "MoE Routing", "Memory"),
    ("bench_topk_sigmoid", "topk_sigmoid", "MoE Routing", "Memory"),
    ("bench_moe_fused_gate", "moe_fused_gate", "MoE Routing", "Memory"),

    # MoE Experts (5)
    ("bench_prepare_moe_input", "prepare_moe_input", "MoE", "Memory"),
    ("bench_scaled_fp4_experts_quant", "scaled_fp4_experts_quant", "MoE", "Memory"),
    ("bench_cutlass_fp4_group_mm", "cutlass_fp4_group_mm", "MoE", "Compute"),
    ("bench_apply_shuffle_mul_sum", "apply_shuffle_mul_sum", "MoE", "Memory"),
    ("bench_moe_align_block_size", "moe_align_block_size", "MoE", "Memory"),

    # Fused MoE (2)
    ("bench_trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_moe", "MoE", "Mixed"),
    ("bench_fused_moe_kernel", "fused_moe_kernel", "MoE", "Mixed"),
]


def reset_cuda_context():
    """Reset CUDA context to clean state."""
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Force garbage collection to release any lingering references
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: CUDA reset failed: {e}")


def run_benchmark(module_name: str, output_dir: str, batch_sizes: str, seq_lens: str) -> bool:
    """Run a single benchmark module."""
    try:
        module = importlib.import_module(module_name)
        batch_list = [int(x) for x in batch_sizes.split(",")]
        seq_list = [int(x) for x in seq_lens.split(",")]

        # Call the run_benchmarks function
        if hasattr(module, 'run_benchmarks'):
            module.run_benchmarks(batch_list, seq_list, output_dir)

        # Clean up CUDA context after each kernel to prevent corruption
        reset_cuda_context()
        return True
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        # Try to reset CUDA context after error
        reset_cuda_context()
        return False


def aggregate_results(output_dir: str):
    """Aggregate all individual CSV results into one file."""
    all_results = []
    headers = None

    for csv_file in sorted(os.listdir(output_dir)):
        if csv_file.endswith('.csv') and csv_file != 'all_kernels.csv':
            csv_path = os.path.join(output_dir, csv_file)
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                file_headers = next(reader)
                if headers is None:
                    headers = file_headers
                for row in reader:
                    all_results.append(row)

    if headers and all_results:
        all_csv_path = os.path.join(output_dir, 'all_kernels.csv')
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_results)
        print(f"\nAggregated results saved to {all_csv_path}")
        print(f"Total benchmark results: {len(all_results)}")


def generate_summary(output_dir: str, kernels_run: List[str], kernels_failed: List[str]):
    """Generate a summary markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = os.path.join(output_dir, 'benchmark_summary.md')

    with open(summary_path, 'w') as f:
        f.write("# DeepSeek-R1-NVFP4-v2 Kernel Benchmark Summary\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Total Kernels:** {len(KERNELS)}\n")
        f.write(f"**Kernels Run:** {len(kernels_run)}\n")
        f.write(f"**Kernels Failed:** {len(kernels_failed)}\n\n")

        f.write("## Kernel Status\n\n")
        f.write("| # | Kernel | Category | Bound | Status |\n")
        f.write("|---|--------|----------|-------|--------|\n")

        for i, (module, kernel, category, bound) in enumerate(KERNELS, 1):
            if kernel in kernels_run:
                status = "OK"
            elif kernel in kernels_failed:
                status = "FAILED"
            else:
                status = "SKIPPED"
            f.write(f"| {i} | `{kernel}` | {category} | {bound} | {status} |\n")

        f.write("\n## CSV Files Generated\n\n")
        for csv_file in sorted(os.listdir(output_dir)):
            if csv_file.endswith('.csv'):
                f.write(f"- `{csv_file}`\n")

    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all DeepSeek kernel benchmarks")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--kernels", type=str, default=None,
                        help="Comma-separated list of kernels to run (default: all)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--list", action="store_true", help="List all available kernels")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable kernels (23 total):\n")
        print(f"{'#':>2}  {'Kernel':<45} {'Category':<12} {'Bound':<8}")
        print("-" * 70)
        for i, (module, kernel, category, bound) in enumerate(KERNELS, 1):
            print(f"{i:>2}  {kernel:<45} {category:<12} {bound:<8}")
        return

    os.makedirs(args.output, exist_ok=True)

    # Determine which kernels to run
    if args.kernels:
        selected = set(args.kernels.split(","))
        kernels_to_run = [(m, k, c, b) for m, k, c, b in KERNELS if k in selected]
    else:
        kernels_to_run = KERNELS

    print("=" * 70)
    print("DeepSeek-R1-NVFP4-v2 Kernel Benchmarks")
    print("=" * 70)
    print(f"Kernels to run: {len(kernels_to_run)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Output directory: {args.output}")
    print("=" * 70)

    kernels_run = []
    kernels_failed = []

    for module_name, kernel_name, category, bound in kernels_to_run:
        print(f"\n{'='*70}")
        print(f"Running: {kernel_name} ({category}, {bound})")
        print(f"{'='*70}")

        success = run_benchmark(module_name, args.output, args.batch_sizes, args.seq_lens)
        if success:
            kernels_run.append(kernel_name)
        else:
            kernels_failed.append(kernel_name)

    # Aggregate results
    aggregate_results(args.output)

    # Generate summary
    generate_summary(args.output, kernels_run, kernels_failed)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print(f"Successful: {len(kernels_run)}/{len(kernels_to_run)}")
    if kernels_failed:
        print(f"Failed: {kernels_failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
