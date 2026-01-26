#!/usr/bin/env python3
"""
Benchmark: fused_moe_kernel (Kernel #23)
Source: triton
Category: MoE (Mixed bound)
Ops: Fused MoE (triton implementation)

Usage:
    python bench_fused_moe_kernel.py --output ../results/

Note: This kernel requires flashinfer to be installed because sglang's
import chain includes flashinfer dependencies.

API from sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:
    fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,           # [num_experts, intermediate_size, hidden_size]
        w2: torch.Tensor,           # [num_experts, hidden_size, intermediate_size]
        topk_output: StandardTopKOutput,  # NamedTuple(topk_weights, topk_ids, router_logits)
        moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
        ...
    ) -> torch.Tensor
"""

import argparse
import sys
from typing import List, Optional, NamedTuple

import torch

from bench_utils import (
    H, E, K, I,
    BenchmarkResult, PEAK_TFLOPS_FP16, PEAK_BANDWIDTH_GBS, RIDGE_FP16,
    compute_gemm_flops,
    benchmark_kernel, save_results, check_sgl_kernel
)

# Try to import fused_moe and related types at module level to fail fast
_fused_moe = None
_StandardTopKOutput = None
_MoeRunnerConfig = None
_import_error = None
try:
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as _fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput as _StandardTopKOutput
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig as _MoeRunnerConfig
except Exception as e:
    _import_error = str(e)


def bench_fused_moe_kernel(B: int, S: int, hidden_size: int,
                           num_experts: int, topk: int, intermediate_size: int,
                           phase: str, device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark triton fused_moe_kernel."""
    global _fused_moe, _StandardTopKOutput, _MoeRunnerConfig, _import_error
    if _fused_moe is None or _StandardTopKOutput is None or _MoeRunnerConfig is None:
        if _import_error:
            print(f"Warning: fused_moe_kernel not available (requires flashinfer: {_import_error})")
        else:
            print("Warning: fused_moe_kernel not available")
        return None

    fused_moe = _fused_moe
    StandardTopKOutput = _StandardTopKOutput
    MoeRunnerConfig = _MoeRunnerConfig

    tokens = B * S if phase == "prefill" else B

    # Hidden states: [tokens, hidden_size]
    hidden_states = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Expert weights (gated MoE: w1 is gate+up fused, w2 is down)
    # w1: [num_experts, 2*intermediate_size, hidden_size] (gate_up fused)
    # w2: [num_experts, hidden_size, intermediate_size] (down)
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)

    # Router outputs - compute topk
    router_logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), topk, dim=-1)

    # Create StandardTopKOutput NamedTuple
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights.to(torch.bfloat16),
        topk_ids=topk_ids.to(torch.int32),
        router_logits=router_logits,
    )

    # Create MoeRunnerConfig
    moe_config = MoeRunnerConfig(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=topk,
        activation="silu",
        is_gated=True,
    )

    def kernel_fn():
        fused_moe(hidden_states, w1, w2, topk_output, moe_config)

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

    # FLOPS: gate_up GEMM (fused) + down GEMM
    # gate_up: [tokens*topk, hidden_size] x [hidden_size, 2*intermediate_size]
    # down: [tokens*topk, intermediate_size] x [intermediate_size, hidden_size]
    total_expert_tokens = tokens * topk
    flops_gate_up = compute_gemm_flops(total_expert_tokens, 2 * intermediate_size, hidden_size)
    flops_down = compute_gemm_flops(total_expert_tokens, hidden_size, intermediate_size)
    flops = flops_gate_up + flops_down

    # Memory
    bytes_input = hidden_states.numel() * 2
    bytes_weights = (w1.numel() + w2.numel()) * 2  # bf16
    bytes_output = hidden_states.numel() * 2
    bytes_transferred = bytes_input + bytes_weights + bytes_output

    gflops = flops / (latency_ms * 1e-3) / 1e9
    tflops = gflops / 1000
    bandwidth_gbs = bytes_transferred / (latency_ms * 1e-3) / 1e9
    arith_intensity = flops / bytes_transferred

    bound = "memory" if arith_intensity < RIDGE_FP16 else "compute"
    if bound == "compute":
        peak_pct = (tflops / PEAK_TFLOPS_FP16) * 100
    else:
        peak_pct = (bandwidth_gbs / PEAK_BANDWIDTH_GBS) * 100

    return BenchmarkResult(
        kernel="fused_moe_kernel",
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
    """Run fused_moe_kernel benchmarks."""
    results = []

    # Decode phase (S=1)
    print("\n=== Decode Phase ===")
    for B in batch_sizes:
        result = bench_fused_moe_kernel(B, 1, H, E, K, I, "decode")
        if result:
            results.append(result)
            print(f"  B={B}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak ({result.bound})")

    # Prefill phase
    print("\n=== Prefill Phase ===")
    for B in batch_sizes[:4]:
        for S in seq_lens:
            result = bench_fused_moe_kernel(B, S, H, E, K, I, "prefill")
            if result:
                results.append(result)
                print(f"  B={B}, S={S}: {result.latency_ms:.4f} ms, {result.gflops:.1f} GFLOPS, {result.peak_pct:.2f}% peak ({result.bound})")

    if results:
        save_results(results, output_dir, "fused_moe_kernel")
    else:
        print("\nNo results - kernel not available")


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused_moe_kernel")
    parser.add_argument("--output", type=str, default="../results/", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048",
                        help="Comma-separated sequence lengths")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 60)
    print("Benchmark: fused_moe_kernel (Kernel #23)")
    print("=" * 60)
    run_benchmarks(batch_sizes, seq_lens, args.output)


if __name__ == "__main__":
    main()
