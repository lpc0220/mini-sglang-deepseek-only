# GPU Cluster Test Command

**Date:** 2026-01-11
**Hardware:** GB200, 4x GPUs
**Model:** DeepSeek-R1-0528-FP4

## Test Script

```bash
#!/bin/bash

INPUT_MEAN=${1:-1024}
OUTPUT_MEAN=${2:-1024}
TP=4
MODEL=/lustre/fsw/coreai_dlfw_dev/pengchengl/DeepSeek-R1-0528-FP4

python -m sglang.bench_one_batch --model-path $MODEL \
  --enable-dp-attention --disable-radix-cache \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend trtllm_mla \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024 --cuda-graph-max-bs 1024 \
  --stream-interval 1 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size=$TP --ep-size=$TP \
  --moe-runner-backend=flashinfer_trtllm \
  --data-parallel-size=$TP \
  --batch-size 2 --input-len 64 --output-len 4
```

## Configuration Details

### Model Settings
- **Model:** DeepSeek-R1-0528-FP4
- **Quantization:** modelopt_fp4 (NVIDIA FP4)
- **KV Cache:** fp8_e4m3

### Attention Backend
- **Backend:** trtllm_mla (NVIDIA TensorRT-LLM MLA)
- **This confirms:** We should remove DeepSeek FlashMLA (using NVIDIA's instead)

### MoE Backend
- **Backend:** flashinfer_trtllm
- **Optimal for:** GB200 Blackwell architecture

### Parallelism
- **Tensor Parallel (TP):** 4 GPUs
- **Data Parallel (DP):** 4 GPUs
- **Expert Parallel (EP):** 4 GPUs
- **DP Attention:** Enabled

### Memory & Performance
- **Memory fraction:** 0.85 (85% GPU memory)
- **Chunked prefill:** 32768 tokens
- **Max prefill:** 32768 tokens
- **CUDA graph batch sizes:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- **Max CUDA graph bs:** 1024

### Test Workload
- **Batch size:** 2
- **Input length:** 64 tokens
- **Output length:** 4 tokens

## Notes

This command uses **NVIDIA's TensorRT-LLM MLA backend** (`--attention-backend trtllm_mla`), which confirms our TODO to remove DeepSeek's FlashMLA is correct - the production deployment uses NVIDIA's implementation.
