#!/usr/bin/env python3
"""
Plot roofline chart for DeepSeek kernels on GB200.

Usage:
    python plot_roofline.py --input ../results/all_kernels.csv --output ../roofline.png
"""

import argparse
import csv
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# GB200 specs
PEAK_TFLOPS_FP4 = 9000
PEAK_TFLOPS_FP8 = 4500
PEAK_TFLOPS_FP16 = 2250
PEAK_BANDWIDTH_GBS = 8000

# Convert to same units (GFLOPS and GB/s)
PEAK_GFLOPS_FP4 = PEAK_TFLOPS_FP4 * 1000
PEAK_GFLOPS_FP8 = PEAK_TFLOPS_FP8 * 1000
PEAK_GFLOPS_FP16 = PEAK_TFLOPS_FP16 * 1000


def load_results(csv_path: str) -> List[Dict]:
    """Load benchmark results from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'kernel': row['kernel'],
                'op': row['op'],
                'phase': row['phase'],
                'B': int(row['B']),
                'S': int(row['S']),
                'gflops': float(row['gflops']),
                'bandwidth_gbs': float(row['bandwidth_gbs']),
                'arith_intensity': float(row['arith_intensity']),
                'bound': row['bound'],
            })
    return results


def plot_roofline(results: List[Dict], output_path: str, title: str = "GB200 Roofline - DeepSeek-R1-NVFP4-v2"):
    """Generate roofline plot."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Arithmetic intensity range
    ai_range = np.logspace(-2, 5, 1000)

    # Roofline curves
    roofline_fp4 = np.minimum(PEAK_GFLOPS_FP4, ai_range * PEAK_BANDWIDTH_GBS)
    roofline_fp8 = np.minimum(PEAK_GFLOPS_FP8, ai_range * PEAK_BANDWIDTH_GBS)
    roofline_fp16 = np.minimum(PEAK_GFLOPS_FP16, ai_range * PEAK_BANDWIDTH_GBS)

    # Plot rooflines
    ax.loglog(ai_range, roofline_fp4, 'b-', linewidth=2, label=f'FP4 Peak ({PEAK_TFLOPS_FP4} TFLOPS)')
    ax.loglog(ai_range, roofline_fp8, 'g-', linewidth=2, label=f'FP8 Peak ({PEAK_TFLOPS_FP8} TFLOPS)')
    ax.loglog(ai_range, roofline_fp16, 'r-', linewidth=2, label=f'FP16 Peak ({PEAK_TFLOPS_FP16} TFLOPS)')

    # Memory bandwidth line
    memory_line = ai_range * PEAK_BANDWIDTH_GBS
    ax.loglog(ai_range, memory_line, 'k--', linewidth=1, alpha=0.5, label=f'Memory BW ({PEAK_BANDWIDTH_GBS} GB/s)')

    # Ridge points
    ridge_fp4 = PEAK_GFLOPS_FP4 / PEAK_BANDWIDTH_GBS
    ridge_fp8 = PEAK_GFLOPS_FP8 / PEAK_BANDWIDTH_GBS
    ridge_fp16 = PEAK_GFLOPS_FP16 / PEAK_BANDWIDTH_GBS

    ax.axvline(x=ridge_fp4, color='b', linestyle=':', alpha=0.5)
    ax.axvline(x=ridge_fp8, color='g', linestyle=':', alpha=0.5)
    ax.axvline(x=ridge_fp16, color='r', linestyle=':', alpha=0.5)

    # Color map for kernels
    kernel_colors = {
        'cutlass_scaled_fp4_mm': 'blue',
        'cutlass_fp4_group_mm': 'cyan',
        'bmm_fp8': 'green',
        'dsv3_fused_a_gemm': 'purple',
        'dsv3_router_gemm': 'magenta',
        'cutlass_mla_decode': 'orange',
        'rmsnorm': 'red',
        'fused_add_rmsnorm': 'darkred',
        'silu_and_mul': 'brown',
        'topk_softmax': 'pink',
    }

    # Marker map for phases
    phase_markers = {
        'decode': 'o',
        'prefill': 's',
    }

    # Plot kernel data points
    plotted_kernels = set()
    for r in results:
        color = kernel_colors.get(r['kernel'], 'gray')
        marker = phase_markers.get(r['phase'], 'x')

        # Use bandwidth-derived performance for memory-bound, gflops for compute-bound
        if r['bound'] == 'memory':
            perf = r['bandwidth_gbs'] * r['arith_intensity']  # Effective GFLOPS
        else:
            perf = r['gflops']

        label = f"{r['kernel']} ({r['phase']})" if r['kernel'] not in plotted_kernels else None
        ax.scatter(r['arith_intensity'], perf, c=color, marker=marker, s=100,
                   alpha=0.7, label=label, edgecolors='black', linewidths=0.5)
        plotted_kernels.add(r['kernel'])

    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_xlim([0.01, 100000])
    ax.set_ylim([1, PEAK_GFLOPS_FP4 * 2])

    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    # Add region labels
    ax.text(0.02, PEAK_GFLOPS_FP16 / 10, 'Memory\nBound', fontsize=10, alpha=0.7)
    ax.text(ridge_fp4 * 10, PEAK_GFLOPS_FP4 * 0.7, 'Compute\nBound', fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved roofline plot to {output_path}")


def plot_roofline_by_phase(results: List[Dict], output_dir: str):
    """Generate separate roofline plots for decode and prefill phases."""
    for phase in ['decode', 'prefill']:
        phase_results = [r for r in results if r['phase'] == phase]
        if phase_results:
            output_path = f"{output_dir}/roofline_{phase}.png"
            plot_roofline(phase_results, output_path,
                         title=f"GB200 Roofline - DeepSeek-R1 {phase.capitalize()} Phase")


def generate_roofline_report(results: List[Dict], output_path: str):
    """Generate markdown report with roofline analysis."""
    # Group by kernel
    kernel_data = {}
    for r in results:
        key = (r['kernel'], r['op'])
        if key not in kernel_data:
            kernel_data[key] = []
        kernel_data[key].append(r)

    with open(output_path, 'w') as f:
        f.write("# GB200 Roofline Analysis - DeepSeek-R1-NVFP4-v2\n\n")

        f.write("## Hardware Limits\n\n")
        f.write(f"- Peak Compute (FP4): {PEAK_TFLOPS_FP4} TFLOPS\n")
        f.write(f"- Peak Compute (FP8): {PEAK_TFLOPS_FP8} TFLOPS\n")
        f.write(f"- Peak Compute (FP16): {PEAK_TFLOPS_FP16} TFLOPS\n")
        f.write(f"- Peak Bandwidth: {PEAK_BANDWIDTH_GBS} GB/s\n")
        f.write(f"- Ridge Point (FP4): {PEAK_GFLOPS_FP4 / PEAK_BANDWIDTH_GBS:.2f} FLOP/byte\n")
        f.write(f"- Ridge Point (FP8): {PEAK_GFLOPS_FP8 / PEAK_BANDWIDTH_GBS:.2f} FLOP/byte\n")
        f.write(f"- Ridge Point (FP16): {PEAK_GFLOPS_FP16 / PEAK_BANDWIDTH_GBS:.2f} FLOP/byte\n\n")

        f.write("## Kernel Placement\n\n")
        f.write("| Kernel | Op | Phase | AI (FLOP/byte) | Bound | GFLOPS | BW (GB/s) |\n")
        f.write("|--------|-----|-------|----------------|-------|--------|----------|\n")

        for (kernel, op), data_list in sorted(kernel_data.items()):
            # Take first entry as representative
            r = data_list[0]
            f.write(f"| {kernel} | {op} | {r['phase']} | {r['arith_intensity']:.2f} | "
                   f"{r['bound']} | {r['gflops']:.0f} | {r['bandwidth_gbs']:.0f} |\n")

        f.write("\n## Bottleneck Analysis\n\n")

        # Decode analysis
        decode_results = [r for r in results if r['phase'] == 'decode']
        if decode_results:
            compute_kernels = [r['kernel'] for r in decode_results if r['bound'] == 'compute']
            memory_kernels = [r['kernel'] for r in decode_results if r['bound'] == 'memory']

            f.write("### Decode Phase\n\n")
            f.write(f"- **Compute-bound kernels**: {', '.join(set(compute_kernels)) or 'None'}\n")
            f.write(f"- **Memory-bound kernels**: {', '.join(set(memory_kernels)) or 'None'}\n\n")

        # Prefill analysis
        prefill_results = [r for r in results if r['phase'] == 'prefill']
        if prefill_results:
            compute_kernels = [r['kernel'] for r in prefill_results if r['bound'] == 'compute']
            memory_kernels = [r['kernel'] for r in prefill_results if r['bound'] == 'memory']

            f.write("### Prefill Phase\n\n")
            f.write(f"- **Compute-bound kernels**: {', '.join(set(compute_kernels)) or 'None'}\n")
            f.write(f"- **Memory-bound kernels**: {', '.join(set(memory_kernels)) or 'None'}\n\n")

    print(f"Saved roofline report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot roofline for DeepSeek kernels")
    parser.add_argument("--input", type=str, default="../results/all_kernels.csv",
                        help="Input CSV file with benchmark results")
    parser.add_argument("--output", type=str, default="../roofline.png",
                        help="Output path for roofline plot")
    parser.add_argument("--report", type=str, default="../roofline.md",
                        help="Output path for roofline markdown report")
    parser.add_argument("--by-phase", action="store_true",
                        help="Generate separate plots for each phase")
    args = parser.parse_args()

    print(f"Loading results from {args.input}")
    results = load_results(args.input)
    print(f"Loaded {len(results)} data points")

    # Generate main roofline plot
    plot_roofline(results, args.output)

    # Generate phase-specific plots if requested
    if args.by_phase:
        output_dir = "/".join(args.output.split("/")[:-1]) or "."
        plot_roofline_by_phase(results, output_dir)

    # Generate markdown report
    generate_roofline_report(results, args.report)


if __name__ == "__main__":
    main()
