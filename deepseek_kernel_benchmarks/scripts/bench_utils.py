#!/usr/bin/env python3
"""
Shared utilities for DeepSeek-R1-NVFP4-v2 kernel benchmarks.

Contains model parameters, hardware specs, and common benchmark functions.
"""

import csv
import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton.testing

# =============================================================================
# DeepSeek-R1-NVFP4-v2 Model Parameters
# =============================================================================
H = 7168       # hidden_size
Nh = 128       # num_heads
Lq = 1536      # q_lora_rank
Lkv = 512      # kv_lora_rank
Dn = 128       # qk_nope_head_dim
Dr = 64        # qk_rope_head_dim
Dv = 128       # v_head_dim
Dq = Dn + Dr   # qk_head_dim = 192
E = 256        # n_routed_experts
K = 8          # num_experts_per_tok
I = 2048       # moe_intermediate_size

# =============================================================================
# GB200 Hardware Specifications
# =============================================================================
PEAK_TFLOPS_FP4 = 9000
PEAK_TFLOPS_FP8 = 4500
PEAK_TFLOPS_FP16 = 2250
PEAK_BANDWIDTH_GBS = 8000

# Ridge points (FLOP/byte)
RIDGE_FP4 = PEAK_TFLOPS_FP4 * 1000 / PEAK_BANDWIDTH_GBS    # 1125
RIDGE_FP8 = PEAK_TFLOPS_FP8 * 1000 / PEAK_BANDWIDTH_GBS    # 562.5
RIDGE_FP16 = PEAK_TFLOPS_FP16 * 1000 / PEAK_BANDWIDTH_GBS  # 281.25

# =============================================================================
# Result Data Class
# =============================================================================
@dataclass
class BenchmarkResult:
    kernel: str
    op: str
    phase: str
    B: int
    S: int
    M: int
    N: int
    K_dim: int
    latency_ms: float
    gflops: float
    peak_pct: float
    bandwidth_gbs: float
    arith_intensity: float
    bound: str  # "compute" or "memory"


# =============================================================================
# Helper Functions
# =============================================================================
def get_dtype_size(dtype) -> int:
    """Get size in bytes for a dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1
    elif dtype == torch.int8:
        return 1
    else:
        return 2  # default


def compute_gemm_flops(M: int, N: int, K: int) -> int:
    """Compute FLOPS for GEMM: 2*M*N*K"""
    return 2 * M * N * K


def compute_bmm_flops(batch: int, M: int, N: int, K: int) -> int:
    """Compute FLOPS for BMM: 2*batch*M*N*K"""
    return 2 * batch * M * N * K


def compute_gemm_bytes(M: int, N: int, K: int, dtype_size: int = 2,
                       weight_dtype_size: float = 0.5) -> int:
    """Compute bytes transferred for GEMM (FP4 weights, FP16 activations)."""
    # Input: M*K * dtype_size
    # Weight: K*N * weight_dtype_size (FP4 = 0.5 bytes)
    # Output: M*N * dtype_size
    return int(M * K * dtype_size + K * N * weight_dtype_size + M * N * dtype_size)


def compute_norm_flops(N: int) -> int:
    """Compute FLOPS for RMSNorm: ~5 ops per element."""
    return 5 * N


def compute_norm_bytes(N: int, dtype_size: int = 2) -> int:
    """Compute bytes for RMSNorm: read + write."""
    return 2 * N * dtype_size


def compute_activation_flops(N: int) -> int:
    """Compute FLOPS for SiLU activation: ~4 ops per element."""
    return 4 * N


def compute_activation_bytes(N: int, dtype_size: int = 2) -> int:
    """Compute bytes for activation: read 2x (gate, up), write 1x."""
    return 3 * N * dtype_size


def benchmark_kernel(kernel_fn, warmup: int = 10, iters: int = 100) -> float:
    """Benchmark a kernel function and return median latency in ms."""
    # Warmup
    for _ in range(warmup):
        kernel_fn()

    torch.cuda.synchronize()

    # Use triton's cudagraph benchmarking for accurate timing
    ms, _, _ = triton.testing.do_bench_cudagraph(kernel_fn, quantiles=[0.5, 0.2, 0.8])
    return ms


def save_results(results: list, output_dir: str, kernel_name: str):
    """Save benchmark results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{kernel_name}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kernel', 'op', 'phase', 'B', 'S', 'M', 'N', 'K',
                        'latency_ms', 'gflops', 'peak_pct', 'bandwidth_gbs',
                        'arith_intensity', 'bound'])
        for r in results:
            writer.writerow([r.kernel, r.op, r.phase, r.B, r.S, r.M, r.N, r.K_dim,
                            f"{r.latency_ms:.6f}", f"{r.gflops:.2f}",
                            f"{r.peak_pct:.2f}", f"{r.bandwidth_gbs:.2f}",
                            f"{r.arith_intensity:.4f}", r.bound])
    print(f"Saved {csv_path}")
    return csv_path


def check_sgl_kernel() -> Optional[object]:
    """Check if sgl_kernel is available."""
    try:
        import sgl_kernel
        return sgl_kernel
    except ImportError:
        print("Warning: sgl_kernel not available")
        return None


def check_flashinfer() -> Optional[object]:
    """Check if flashinfer is available."""
    try:
        import flashinfer
        return flashinfer
    except ImportError:
        print("Warning: flashinfer not available")
        return None
