#!/usr/bin/env python3
"""
Benchmark: trtllm_fp4_block_scale_moe (Kernel #22)
Source: flashinfer (TensorRT-LLM kernel)
Category: MoE (Mixed bound)
Ops: Fused MoE with FP4 block-scaled weights

Based on: flashinfer/benchmarks/routines/moe.py testTrtllmFp4BlockScaleMoe

Usage:
    python bench_trtllm_fp4_block_scale_moe.py --output ../results/
"""

import argparse
import multiprocessing as mp
from typing import List, Optional

import torch

from bench_utils import (
    H, E, K, I,
    BenchmarkResult, PEAK_TFLOPS_FP4, PEAK_BANDWIDTH_GBS, RIDGE_FP4,
    compute_gemm_flops,
    benchmark_kernel, save_results, check_flashinfer
)

# DeepSeek MoE configuration
N_GROUP = 8  # Number of expert groups for DeepSeek routing
TOPK_GROUP = 4  # Number of groups for top-k routing
ROUTED_SCALING_FACTOR = 2.5  # DeepSeek V3 scaling factor
ROUTING_METHOD_DEEPSEEK_V3 = 2  # DeepSeek V3 routing method type


def calculate_fp4_global_scale_factor(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate global scale factor for FP4 quantization."""
    tensor_amax = tensor.abs().max().to(torch.float32)
    # FLOAT8_E4M3_MAX = 448, FLOAT4_E2M1_MAX = 6
    global_scale = (448.0 * 6.0) / tensor_amax.clamp(min=1e-12)
    return global_scale


def quant_fp4_simple(a: torch.Tensor, a_global_sf: torch.Tensor,
                     use_ue8m0: bool = False, is_sf_swizzled_layout: bool = True):
    """
    FP4 quantization for benchmarking.
    Uses flashinfer's fp4_quantize function.
    """
    from flashinfer import fp4_quantize
    sf_vec_size = 16
    a_fp4, a_sf = fp4_quantize(a, a_global_sf, sf_vec_size, use_ue8m0, is_sf_swizzled_layout)
    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches_simple(a: torch.Tensor, num_experts: int,
                              use_ue8m0: bool = False, is_sf_swizzled_layout: bool = True):
    """FP4 batch quantization for benchmarking."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_global_sf = calculate_fp4_global_scale_factor(a[i])
        a_fp4, a_sf, _ = quant_fp4_simple(a[i], a_global_sf, use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def bench_trtllm_fp4_block_scale_moe(flashinfer, B: int, S: int, hidden_size: int,
                                      num_experts: int, topk: int, intermediate_size: int,
                                      phase: str, device: str = "cuda") -> Optional[BenchmarkResult]:
    """Benchmark TRT-LLM FP4 block-scaled MoE kernel.

    Based on flashinfer/tests/moe/test_trtllm_gen_routed_fused_moe.py:
        - Uses NvFP4 quantization mode (sf_vec_size=16, use_ue8m0=False, is_sf_swizzled_layout=False for input)
        - Uses routing_logits, not pre-computed expert indices
        - DeepSeek V3 routing method with n_group=8, topk_group=4
    """
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
        from flashinfer import fp4_quantize
    except ImportError:
        print("Warning: trtllm_fp4_block_scale_moe not available (requires flashinfer)")
        return None

    tokens = B * S if phase == "prefill" else B
    local_num_experts = num_experts
    local_expert_offset = 0

    torch.manual_seed(42)

    # Routing logits: [tokens, num_experts] - bfloat16 for non-DeepSeekV3, float32 for DeepSeekV3
    # But the test uses bfloat16 for routing_logits
    routing_logits = torch.rand(tokens, num_experts, dtype=torch.bfloat16, device=device)
    # DeepSeek V3 uses routing bias
    routing_bias = torch.zeros(num_experts, dtype=torch.bfloat16, device=device)

    # Hidden states: [tokens, hidden_size] - start with bfloat16 for quantization
    # Match test: use * 0.1 to scale down values
    hidden_states_bf16 = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.1

    # Create weights: [num_experts, 2 * intermediate_size, hidden_size] for gemm1
    #                 [num_experts, hidden_size, intermediate_size] for gemm2
    # Match test: use * 0.1 to scale down values
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        dtype=torch.bfloat16, device=device
    ) * 0.1
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.bfloat16, device=device
    ) * 0.1

    # FP4 quantization setup - match test configuration for NvFP4
    # Use fixed global scale factor as in test: 448.0 * 6.0 = 2688.0
    fixed_global_scale = torch.tensor([448.0 * 6.0], device=device)
    global_scale_inv = 1.0 / 448.0 / 6.0  # For output scaling

    # Quantize hidden states with is_sf_swizzled_layout=False (match test)
    hidden_states_fp4_bytes, hidden_states_scale_fp4_bytes = fp4_quantize(
        hidden_states_bf16,
        fixed_global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,  # Match test!
    )
    hidden_states_fp4 = hidden_states_fp4_bytes.view(torch.uint8).reshape(
        tokens, hidden_size // 2
    )
    hidden_states_scale_linear_fp4 = hidden_states_scale_fp4_bytes.view(torch.float8_e4m3fn).reshape(
        tokens, -1
    )

    # Quantize weights with is_sf_swizzled_layout=True (default for weights)
    gemm1_weights_fp4_bytes, gemm1_weights_scale_bytes = fp4_quantize(
        gemm1_weights.reshape(-1, gemm1_weights.shape[-1]),  # Flatten experts for quantization
        fixed_global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    # Reshape back
    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(torch.uint8).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    gemm1_weights_scale = gemm1_weights_scale_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, -1
    )

    gemm2_weights_fp4_bytes, gemm2_weights_scale_bytes = fp4_quantize(
        gemm2_weights.reshape(-1, gemm2_weights.shape[-1]),
        fixed_global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(torch.uint8).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    gemm2_weights_scale = gemm2_weights_scale_bytes.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, -1
    )

    # Optional parameters (None for benchmarking)
    gemm1_bias = None
    gemm1_alpha = None
    gemm1_beta = None
    gemm1_clamp_limit = None
    gemm2_bias = None

    # Create scale scalars - use global_scale_inv for proper dequantization
    # Match test: output1_scale = hidden_states_global_scale * w13_global_scale
    output1_scale_scalar = torch.tensor(
        [global_scale_inv * global_scale_inv] * num_experts, device=device, dtype=torch.float32
    )
    output1_scale_gate_scalar = torch.tensor(
        [global_scale_inv * global_scale_inv] * num_experts, device=device, dtype=torch.float32
    )
    output2_scale_scalar = torch.tensor(
        [global_scale_inv * global_scale_inv] * num_experts, device=device, dtype=torch.float32
    )

    def kernel_fn():
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states_fp4,
            hidden_states_scale=hidden_states_scale_linear_fp4,
            gemm1_weights=gemm1_weights_fp4,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights_fp4,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            output2_scale_scalar=output2_scale_scalar,
            num_experts=num_experts,
            top_k=topk,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            routing_method_type=ROUTING_METHOD_DEEPSEEK_V3,
            gated_act_type=0,  # SwiGLU
            do_finalize=True,
        )

    try:
        latency_ms = benchmark_kernel(kernel_fn)
    except Exception as e:
        error_str = str(e)
        print(f"Warning: Kernel failed for B={B}, S={S}: {error_str}")
        # Try to recover CUDA context
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Force CUDA to reinitialize by creating a small tensor
            _ = torch.zeros(1, device="cuda")
        except:
            pass
        return None

    # FLOPS: gate_up GEMM + down GEMM for each expert token
    total_expert_tokens = tokens * topk
    flops_gate_up = compute_gemm_flops(total_expert_tokens, 2 * intermediate_size, hidden_size)
    flops_down = compute_gemm_flops(total_expert_tokens, hidden_size, intermediate_size)
    flops = flops_gate_up + flops_down

    # Memory: inputs (FP4), weights (FP4), outputs (BF16)
    bytes_input = tokens * hidden_size * 0.5  # FP4
    bytes_weights = (
        num_experts * 2 * intermediate_size * hidden_size * 0.5 +  # gemm1 FP4
        num_experts * hidden_size * intermediate_size * 0.5  # gemm2 FP4
    )
    bytes_output = tokens * hidden_size * 2  # BF16
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


def _run_single_benchmark(B: int, S: int, hidden_size: int, num_experts: int,
                          topk: int, intermediate_size: int, phase: str, queue: mp.Queue):
    """Run a single benchmark in a subprocess and put result in queue."""
    try:
        flashinfer = check_flashinfer()
        if not flashinfer:
            queue.put(None)
            return
        result = bench_trtllm_fp4_block_scale_moe(flashinfer, B, S, hidden_size,
                                                   num_experts, topk, intermediate_size, phase)
        queue.put(result)
    except Exception as e:
        print(f"Warning: Subprocess failed for B={B}, S={S}: {e}")
        queue.put(None)


def run_benchmark_isolated(B: int, S: int, hidden_size: int, num_experts: int,
                           topk: int, intermediate_size: int, phase: str,
                           timeout: int = 60) -> Optional[BenchmarkResult]:
    """Run a single benchmark in an isolated subprocess to prevent CUDA context corruption."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_single_benchmark,
                      args=(B, S, hidden_size, num_experts, topk, intermediate_size, phase, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"Warning: Benchmark timed out for B={B}, S={S}")
        return None

    if proc.exitcode != 0:
        print(f"Warning: Subprocess crashed for B={B}, S={S} (exit code {proc.exitcode})")
        return None

    try:
        return queue.get_nowait()
    except:
        return None


def run_flashinfer_benchmark(num_tokens: int, hidden_size: int, intermediate_size: int,
                              num_experts: int, top_k: int) -> Optional[float]:
    """Run flashinfer's own benchmark for trtllm_fp4_block_scale_moe.

    Returns median time in ms if successful, None if failed.
    """
    import subprocess
    import sys

    # Find flashinfer benchmark script
    flashinfer_bench = None
    for path in [
        "/Users/lpc/workspace/sglang-deepseek-only/flashinfer/benchmarks/flashinfer_benchmark.py",
        "../../flashinfer/benchmarks/flashinfer_benchmark.py",
    ]:
        import os
        if os.path.exists(path):
            flashinfer_bench = path
            break

    if not flashinfer_bench:
        return None

    cmd = [
        sys.executable, flashinfer_bench,
        "--routine", "trtllm_fp4_block_scale_moe",
        "--num_tokens", str(num_tokens),
        "--hidden_size", str(hidden_size),
        "--intermediate_size", str(intermediate_size),
        "--num_experts", str(num_experts),
        "--top_k", str(top_k),
        "--n_group", str(N_GROUP),
        "--topk_group", str(TOPK_GROUP),
        "--routed_scaling_factor", str(ROUTED_SCALING_FACTOR),
        "--routing_method", "deepseek_v3",
        "--num_iters", "10",
        "--dry_run_iters", "3",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Parse output for median time
            for line in result.stdout.split('\n'):
                if 'median' in line.lower() or 'time' in line.lower():
                    print(f"  flashinfer output: {line}")
            return None  # For now, just report what flashinfer says
        else:
            print(f"  flashinfer benchmark failed: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  flashinfer benchmark error: {e}")
        return None


def run_benchmarks(batch_sizes: List[int], seq_lens: List[int], output_dir: str):
    """Run trtllm_fp4_block_scale_moe benchmarks using flashinfer's benchmark tool."""
    flashinfer = check_flashinfer()
    if not flashinfer:
        print("ERROR: flashinfer not available")
        return

    print("\nNote: This kernel requires complex weight preprocessing (shuffling, permutation)")
    print("that is difficult to replicate outside flashinfer's test framework.")
    print("Using flashinfer's own benchmark infrastructure...\n")

    # Try to run flashinfer's benchmark directly
    import subprocess
    import sys
    import os

    # Find flashinfer benchmark directory - need to run from there for relative imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flashinfer_bench_dir = os.path.join(script_dir, "../../flashinfer/benchmarks")
    if not os.path.exists(flashinfer_bench_dir):
        flashinfer_bench_dir = None

    if not flashinfer_bench_dir:
        print("ERROR: Cannot find flashinfer benchmarks directory")
        print("Please clone flashinfer and run the benchmark directly:")
        print(f"  git clone https://github.com/flashinfer-ai/flashinfer.git")
        print(f"  cd flashinfer/benchmarks")
        print(f"  python flashinfer_benchmark.py --routine trtllm_fp4_block_scale_moe \\")
        print(f"    --num_tokens <tokens> --hidden_size {H} --intermediate_size {I} \\")
        print(f"    --num_experts {E} --top_k {K} --n_group {N_GROUP} --topk_group {TOPK_GROUP} \\")
        print(f"    --routed_scaling_factor {ROUTED_SCALING_FACTOR} --routing_method deepseek_v3")
        return

    results = []

    # Test configurations matching flashinfer's test parameters
    test_configs = [
        # (num_tokens, description)
        (8, "decode B=8"),
        (128, "prefill B=1,S=128"),
        (256, "prefill B=1,S=256"),
        (1024, "prefill B=1,S=1024"),
    ]

    print("=== Running flashinfer benchmark ===")
    print(f"  (running from: {flashinfer_bench_dir})")
    for num_tokens, desc in test_configs:
        # Run from flashinfer/benchmarks directory so relative imports work
        cmd = [
            sys.executable, "flashinfer_benchmark.py",
            "--routine", "trtllm_fp4_block_scale_moe",
            "--num_tokens", str(num_tokens),
            "--hidden_size", str(H),
            "--intermediate_size", str(I),
            "--num_experts", str(E),
            "--top_k", str(K),
            "--n_group", str(N_GROUP),
            "--topk_group", str(TOPK_GROUP),
            "--routed_scaling_factor", str(ROUTED_SCALING_FACTOR),
            "--routing_method", "deepseek_v3",
            "--num_iters", "10",
            "--dry_run_iters", "5",
            "--verbose", "0",
        ]

        print(f"\n  {desc} (tokens={num_tokens}):")
        try:
            # Run from flashinfer/benchmarks directory
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=flashinfer_bench_dir)
            if result.returncode == 0:
                # Parse output for performance metrics
                for line in result.stdout.split('\n'):
                    if 'trtllm' in line.lower() or 'tflops' in line.lower() or 'median' in line.lower():
                        print(f"    {line.strip()}")
            else:
                print(f"    FAILED: {result.stderr[:200] if result.stderr else 'Unknown error'}")
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print("Note: This kernel benchmark uses flashinfer's internal benchmark tool.")
    print("Results may differ from other kernels due to different measurement methods.")
    print("=" * 60)


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
