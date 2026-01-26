#!/usr/bin/env python3
"""
Run all DeepSeek-R1-NVFP4-v2 kernel benchmarks.

This script orchestrates running all 23 kernel benchmarks and aggregates results.
Each benchmark runs in a SEPARATE SUBPROCESS to prevent CUDA context corruption
from affecting other benchmarks.

Usage:
    cd deepseek_kernel_benchmarks/scripts
    python run_all_benchmarks.py --output ../results/
    python run_all_benchmarks.py --kernels rmsnorm,cutlass_scaled_fp4_mm --output ../results/
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

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
    ("bench_concat_mla_mha_k", "concat_mla_k", "Concat", "Memory"),

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


def run_benchmark_subprocess(module_name: str, output_dir: str, batch_sizes: str, seq_lens: str) -> Tuple[bool, str]:
    """Run a single benchmark in a separate subprocess for isolation.

    This ensures that if one kernel crashes and corrupts the CUDA context,
    it won't affect subsequent benchmarks.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, f"{module_name}.py")

    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"

    cmd = [
        sys.executable,
        script_path,
        "--output", output_dir,
        "--batch-sizes", batch_sizes,
        "--seq-lens", seq_lens,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per kernel
        )

        # Print output for visibility
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            return False, f"Exit code {result.returncode}"
        return True, "OK"

    except subprocess.TimeoutExpired:
        return False, "Timeout (10 minutes)"
    except Exception as e:
        return False, str(e)


def aggregate_results(output_dir: str):
    """Aggregate all individual CSV results into one file."""
    all_results = []
    headers = None

    for csv_file in sorted(os.listdir(output_dir)):
        if csv_file.endswith('.csv') and csv_file != 'all_kernels.csv':
            csv_path = os.path.join(output_dir, csv_file)
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    file_headers = next(reader)
                    if headers is None:
                        headers = file_headers
                    for row in reader:
                        all_results.append(row)
            except Exception as e:
                print(f"Warning: Failed to read {csv_file}: {e}")

    if headers and all_results:
        all_csv_path = os.path.join(output_dir, 'all_kernels.csv')
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_results)
        print(f"\nAggregated results saved to {all_csv_path}")
        print(f"Total benchmark results: {len(all_results)}")


def generate_summary(output_dir: str, results: List[Tuple[str, str, str, str, bool, str]]):
    """Generate a summary markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = os.path.join(output_dir, 'benchmark_summary.md')

    successful = sum(1 for r in results if r[4])
    failed = len(results) - successful

    with open(summary_path, 'w') as f:
        f.write("# DeepSeek-R1-NVFP4-v2 Kernel Benchmark Summary\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Total Kernels:** {len(KERNELS)}\n")
        f.write(f"**Kernels Run:** {len(results)}\n")
        f.write(f"**Successful:** {successful}\n")
        f.write(f"**Failed:** {failed}\n\n")

        f.write("**Note:** Each kernel runs in a separate subprocess for isolation.\n")
        f.write("CUDA crashes in one kernel do not affect other kernels.\n\n")

        f.write("## Kernel Status\n\n")
        f.write("| # | Kernel | Category | Bound | Status | Notes |\n")
        f.write("|---|--------|----------|-------|--------|-------|\n")

        for i, (module, kernel, category, bound, success, notes) in enumerate(results, 1):
            status = "✓ OK" if success else "✗ FAILED"
            notes_short = notes[:50] + "..." if len(notes) > 50 else notes
            f.write(f"| {i} | `{kernel}` | {category} | {bound} | {status} | {notes_short} |\n")

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
    print("")
    print("NOTE: Each kernel runs in a separate subprocess for isolation.")
    print("      CUDA crashes in one kernel will NOT affect other kernels.")
    print("=" * 70)

    results = []

    for module_name, kernel_name, category, bound in kernels_to_run:
        print(f"\n{'='*70}")
        print(f"Running: {kernel_name} ({category}, {bound})")
        print(f"{'='*70}")

        success, notes = run_benchmark_subprocess(
            module_name, args.output, args.batch_sizes, args.seq_lens
        )
        results.append((module_name, kernel_name, category, bound, success, notes))

        if success:
            print(f"[OK] {kernel_name} completed successfully")
        else:
            print(f"[FAILED] {kernel_name}: {notes}")

    # Aggregate results
    aggregate_results(args.output)

    # Generate summary
    generate_summary(args.output, results)

    # Print final summary
    successful = sum(1 for r in results if r[4])
    failed = len(results) - successful

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print(f"Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"Failed kernels:")
        for module, kernel, cat, bound, success, notes in results:
            if not success:
                print(f"  - {kernel}: {notes}")
    print("=" * 70)


if __name__ == "__main__":
    main()
