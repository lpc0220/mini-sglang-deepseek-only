# Deletion Completion Analysis Tool

When user says "run deletion completion analysis", follow these steps.

**Reference:** See `tools/registry.md` for the complete kept/removed list.

---

## Step 1: Verify REMOVED Items Have Zero References

For each item in `registry.md` REMOVED section, run the verification command below.
All counts should be **0**.

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang/python
```

### Platform Backends (should all be 0)
```bash
# NPU/Ascend
grep -rn 'NpuCommunicator\|SGLANG_NPU_\|MHA_NPU\|MLA_NPU\|DSA_NPU' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# AMD/ROCm/HIP
grep -rn 'SGLANG_ROCM_\|SGLANG_USE_AITER\|rpd_profiler\|HipCommunicator' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# Intel XPU
grep -rn 'XpuCommunicator' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# Habana HPU
grep -rn 'HpuCommunicator' --include='*.py' sglang/ | grep -v __pycache__ | wc -l
```

### Attention Backends (should all be 0)
```bash
# FA3/FA4
grep -rn 'from sgl_kernel.flash_attn\|from sgl_kernel import.*flash_attn' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# FlashMLA
grep -rn 'from sgl_kernel.flash_mla\|from sgl_kernel import.*flash_mla' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# FlashAttention Backend
grep -rn 'FlashAttentionBackend' --include='*.py' sglang/ | grep -v __pycache__ | wc -l
```

### Models (should all be 0)
```bash
# Mamba/SSM
grep -rn 'mamba' -i --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# GGUF
grep -rn 'gguf' -i --include='*.py' sglang/ | grep -v __pycache__ | wc -l
```

### Infrastructure (should all be 0)
```bash
# DLLM
grep -rn 'dllm\|DLLM\|is_dllm' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# gRPC
grep -rn 'grpc_mode\|grpc_server' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# Transformers backend
grep -rn 'TransformersForCausalLM\|ModelImpl.TRANSFORMERS' --include='*.py' sglang/ | grep -v __pycache__ | wc -l

# MindSpore
grep -rn 'mindspore_runner\|ModelImpl.MINDSPORE' --include='*.py' sglang/ | grep -v __pycache__ | wc -l
```

---

## Step 2: Verify KEPT Items Still Exist

For each item in `registry.md` KEPT section, verify the file/directory exists.

```bash
cd /Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt

# Models
ls models/deepseek.py models/deepseek_v2.py models/deepseek_nextn.py models/deepseek_common/

# Quantization (spot check)
ls layers/quantization/fp8.py layers/quantization/mxfp4.py layers/quantization/gptq.py

# Attention (spot check)
ls layers/attention/flashinfer_backend.py layers/attention/cutlass_mla_backend.py layers/attention/nsa_backend.py

# MoE
ls layers/moe/cutlass_moe.py layers/moe/fused_moe_triton/

# Function call
ls function_call/deepseekv3_detector.py function_call/base_format_detector.py

# Memory cache
ls mem_cache/radix_cache.py mem_cache/memory_pool.py

# Distributed
ls distributed/parallel_state.py distributed/communication_op.py

# Managers
ls managers/scheduler.py managers/tp_worker.py managers/tokenizer_manager.py

# Entrypoints
ls entrypoints/engine.py entrypoints/http_server.py entrypoints/openai/
```

---

## Step 3: If Non-Zero Count Found

If any REMOVED item has count > 0:

```bash
# Show the actual matches
grep -rn "PATTERN" --include="*.py" sglang/ | grep -v __pycache__
```

Then remove the remaining references using Edit tool.

---

## Step 4: Syntax Verification

After any fixes:
```bash
python -m py_compile sglang/srt/path/to/fixed_file.py
```
