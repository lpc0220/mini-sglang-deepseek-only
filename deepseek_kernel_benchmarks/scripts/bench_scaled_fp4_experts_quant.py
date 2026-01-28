#!/usr/bin/env python3
"""
Benchmark: scaled_fp4_experts_quant (Kernel #18)
Source: sgl-kernel
Category: MoE (Memory-bound)
Ops: Quantize expert inputs to FP4 with per-expert scaling

API from sgl_kernel/gemm.py:
    scaled_fp4_experts_quant(
        input_tensor: torch.Tensor,      # [m, k] - expert inputs
        input_global_scale: torch.Tensor, # scalar
        expert_offsets: torch.Tensor,     # [num_experts + 1]
        blockscale_offsets: torch.Tensor, # [num_experts + 1]
        topk: int,
        expert_map: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]

Usage:
    python bench_scaled_fp4_experts_quant.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K,
    BenchmarkResult, PEAK_BANDWIDTH_GBS,
    benchmark_kernel, save_results, check_sgl_kernel
)


def bench_scaled_fp4_experts_quant(sgl_kernel, B: int, S: int, hidden_size: int,
                                    num_experts: int, topk: int, phase: str,
                                    device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark scaled_fp4_experts_quant kernel."""
    try:
        from sgl_kernel import scaled_fp4_experts_quant
    except ImportError:
        print("Warning: scaled_fp4_experts_quant not available (kernel may be internal)")
        return None

    tokens = B * S if phase == "prefill" else B
    total_expert_tokens = tokens * topk

    # Expert inputs: [total_expert_tokens, hidden_size]
    expert_inputs = torch.randn(total_expert_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Calculate global scale factor for FP4 quantization
    # API expects input_global_scale to be 1D tensor with shape [1]
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    input_amax = expert_inputs.abs().max().to(torch.float32)
    input_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / input_amax.clamp(min=1e-12)).reshape(1)

    # Expert offsets: [num_experts + 1] - cumulative token counts per expert
    # For simplicity, distribute tokens evenly across experts
    tokens_per_expert = total_expert_tokens // num_experts
    expert_offsets = torch.arange(0, num_experts + 1, dtype=torch.int32, device=device) * tokens_per_expert
    expert_offsets[-1] = total_expert_tokens  # Fix rounding

    # Blockscale offsets: similar structure for scale factors
    # Each token has hidden_size/16 scale factors (16 elements per block)
    sf_per_token = hidden_size // 16
    blockscale_offsets = torch.arange(0, num_experts + 1, dtype=torch.int32, device=device) * (tokens_per_expert * sf_per_token)
    blockscale_offsets[-1] = total_expert_tokens * sf_per_token

    def kernel_fn():
        scaled_fp4_experts_quant(
            expert_inputs,
            input_global_scale,
            expert_offsets,
            blockscale_offsets,
            topk,
            expert_map=None,
        )

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        print(f"Warning: Kernel failed for B={B}, S={S}: {e}")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
        return None

    # Memory: read bf16 inputs, write fp4 outputs + scales
    bytes_read = expert_inputs.numel() * 2  # bf16
    bytes_write = expert_inputs.numel() // 2 + total_expert_tokens * sf_per_token  # fp4 + fp8 scales
    bytes_transferred = bytes_read + bytes_write
    flops = expert_inputs.numel() * 2  # quantization ops
    gflops = flops / (latency_ms * 1e-3) / 1e9
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="scaled_fp4_experts_quant",
        op="experts_quant",
        phase=phase,
        B=B, S=S,
        M=total_expert_tokens, N=hidden_size, K_dim=0,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound="memory"
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run scaled_fp4_experts_quant benchmarks."""
    sgl_kernel = check_sgl_kernel()
    if not sgl_kernel:
        print("ERROR: sgl_kernel not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_scaled_fp4_experts_quant(sgl_kernel, B, 1, H, E, K, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_scaled_fp4_experts_quant(sgl_kernel, B, S, H, E, K, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.bandwidth_gbs:.1f} GB/s, {result.peak_pct:.1f}% peak")

    if results:
        save_results(results, output_dir, "scaled_fp4_experts_quant")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark scaled_fp4_experts_quant kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: scaled_fp4_experts_quant (Kernel #18)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
