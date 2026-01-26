#!/usr/bin/env python3
"""
Benchmark: trtllm_fp4_block_scale_moe (Kernel #22)
Source: flashinfer (TensorRT-LLM kernel)
Category: MoE (Mixed bound)
Ops: Fused MoE with FP4 block-scaled weights

Usage:
    python bench_trtllm_fp4_block_scale_moe.py --output ../results/
"""

import argparse
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K, I,
    BenchmarkResult, PEAK_TFLOPS_FP4, PEAK_BANDWIDTH_GBS, RIDGE_FP4,
    compute_gemm_flops,
    benchmark_kernel, save_results, check_flashinfer
)


def bench_trtllm_fp4_block_scale_moe(flashinfer, B: int, S: int, hidden_size: int,
                                      num_experts: int, topk: int, intermediate_size: int,
                                      phase: str, device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TRT-LLM FP4 block-scaled MoE kernel."""
    try:
        from flashinfer.triton.moe import trtllm_fp4_block_scale_moe
    except ImportError:
        print("Warning: trtllm_fp4_block_scale_moe not available")
        return None

    tokens = B * S if phase == "prefill" else B

    # Hidden states: [tokens, hidden_size]
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert routing
    expert_indices = torch.randint(0, num_experts, (tokens, topk), dtype=torch.int32, device=device)
    expert_weights = torch.randn(tokens, topk, dtype=torch.float32, device=device)
    expert_weights = torch.softmax(expert_weights, dim=-1)

    # FP4 weights (simplified - actual format is more complex)
    # gate_up: [num_experts, hidden_size, 2*intermediate_size]
    # down: [num_experts, intermediate_size, hidden_size]
    gate_up_weight = torch.randn(num_experts, hidden_size, 2 * intermediate_size,
                                  dtype=torch.bfloat16, device=device)
    down_weight = torch.randn(num_experts, intermediate_size, hidden_size,
                               dtype=torch.bfloat16, device=device)

    def kernel_fn():
        trtllm_fp4_block_scale_moe(
            hidden_states, gate_up_weight, down_weight,
            expert_indices, expert_weights
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

    # FLOPS: gate_up GEMM + down GEMM for each expert token
    total_expert_tokens = tokens * topk
    flops_gate_up = compute_gemm_flops(total_expert_tokens, 2 * intermediate_size, hidden_size)
    flops_down = compute_gemm_flops(total_expert_tokens, hidden_size, intermediate_size)
    flops = flops_gate_up + flops_down

    # Memory: inputs, weights, outputs
    bytes_input = hidden_states.numel() * 2
    bytes_weights = (gate_up_weight.numel() + down_weight.numel()) * 0.5  # FP4
    bytes_output = hidden_states.numel() * 2
    bytes_transferred = int(bytes_input + bytes_weights + bytes_output)

    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    bound = "memory" if arith_intensity < RIDGE_FP4 else "compute"
    if bound == "compute":
        peak_pct = (tflops / PEAK_TFLOPS_FP4) * 100
    else:
        peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="trtllm_fp4_block_scale_moe",
        op="fused_moe",
        phase=phase,
        B=B, S=S,
        M=total_expert_tokens, N=intermediate_size, K_dim=hidden_size,
        latency_ms=latency_ms,
        gflops=gflops,
        peak_pct=peak_pct,
        bandwidth_gbs=bandwidth_gbs,
        arith_intensity=arith_intensity,
        bound=bound
    )


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run trtllm_fp4_block_scale_moe benchmarks."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_trtllm_fp4_block_scale_moe(flashinfer, B, 1, H, E, K, I, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak ({result.bound})")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_trtllm_fp4_block_scale_moe(flashinfer, B, S, H, E, K, I, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak ({result.bound})")

    if results:
        save_results(results, output_dir, "trtllm_fp4_block_scale_moe")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark trtllm_fp4_block_scale_moe kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: trtllm_fp4_block_scale_moe (Kernel #22)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
