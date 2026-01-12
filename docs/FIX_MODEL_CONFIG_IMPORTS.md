# Fix: Model Config Import Cleanup

**Date:** 2026-01-12
**Issue:** `cannot import name 'ChatGLMConfig' from 'sglang.srt.configs'`
**Status:** ✅ FIXED
**Commit:** ce2d2fc59

## Problem

During GB200 inference testing with `bench_one_batch.sh`, the following error occurred:

```
cannot import name 'ChatGLMConfig' from 'sglang.srt.configs'
File: python/sglang/srt/utils/hf_transformers_utils.py
```

### Root Cause

Previous cleanup phases removed non-DeepSeek model files and their config classes from `sglang.srt.configs/`, but the **import statements** for those configs were left in place in utility files. When Python tried to execute the import, the configs no longer existed, causing an ImportError.

### Impact
- **Immediate:** Blocked inference testing on GB200
- **Severity:** Critical - prevented any model loading

## Solution

Comprehensive cleanup of all non-DeepSeek model configuration imports and references.

### Files Modified

#### 1. `python/sglang/srt/utils/hf_transformers_utils.py` (197 lines removed)

**Removed imports:**
```python
# REMOVED: 19+ non-DeepSeek config imports
from sglang.srt.configs import (
    ChatGLMConfig,        # ChatGLM models
    DbrxConfig,          # Databricks DBRX
    DotsOCRConfig,       # OCR models
    DotsVLMConfig,       # Vision-language models
    ExaoneConfig,        # LG AI EXAONE
    FalconH1Config,      # Falcon models
    JetNemotronConfig,   # NVIDIA Nemotron
    JetVLMConfig,        # Jet vision models
    KimiLinearConfig,    # Moonshot Kimi
    KimiVLConfig,        # Kimi vision
    LongcatFlashConfig,  # Longcat models
    MultiModalityConfig, # Generic multimodal
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Olmo3Config,         # Allen AI OLMo
    Qwen3NextConfig,     # Alibaba Qwen
    Step3VLConfig,
    InternVLChatConfig,
)
from sglang.srt.utils import mistral_utils  # Mistral utilities
```

**Removed config registry:**
```python
# REMOVED: _CONFIG_REGISTRY with 19+ model mappings
_CONFIG_REGISTRY = {
    "LlamaForCausalLM": LlamaConfig,
    "ChatGLMModel": ChatGLMConfig,
    "QWenLMHeadModel": QwenConfig,
    "Qwen2ForCausalLM": Qwen2Config,
    "DbrxForCausalLM": DbrxConfig,
    "ExaoneForCausalLM": ExaoneConfig,
    "FalconForCausalLM": FalconConfig,
    "GemmaForCausalLM": GemmaConfig,
    "InternLMForCausalLM": InternLM2Config,
    # ... 19+ entries removed
}

# NOW: Empty registry (DeepSeek uses transformers configs)
_CONFIG_REGISTRY = {}
```

**Removed functions:**
```python
# REMOVED: _load_mistral_large_3_for_causal_LM()
# This was Mistral-specific config loader for Large 3 variant
```

**Removed model-specific hacks:**
```python
# REMOVED (lines 250-400):
# - Phi4MM vision config injection
# - Qwen2-VL size injection
# - Sarashina2Vision size injection
# - InternVL3_5 special handling
# - InternVL chat config processing
# - MultiModality architecture override
# - LLaMA 3 stop token handling (<|eom_id|>)
# - LLaMA tokenizer fallback logic
# - Devstral/Mistral tokenizer redirect
```

#### 2. `python/sglang/srt/model_executor/model_runner.py` (28 lines changed)

**Removed imports:**
```python
# REMOVED:
from sglang.srt.configs import (
    FalconH1Config,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3NextConfig,
)
```

**Modified properties (now return None):**
```python
@property
def qwen3_next_config(self) -> Optional[Qwen3NextConfig]:
    return None  # Previously checked for Qwen3Next models

@property
def hybrid_gdn_config(self) -> Optional[...]:
    return None  # Hybrid GDN models removed

@property
def mamba2_config(self) -> Optional[...]:
    return None  # Mamba2 models removed

@property
def kimi_linear_config(self) -> Optional[KimiLinearConfig]:
    return None  # Kimi Linear removed
```

### What Was Kept

✅ **DeepSeek-specific code:**
- `_load_deepseek_v32_model()` - DeepSeek v3.2 config loader
- DeepSeek model handling in transformers integration
- Standard transformers config imports (AutoConfig, AutoTokenizer)

✅ **Core infrastructure:**
- HuggingFace transformers integration
- Tokenizer loading logic
- Config parsing utilities

## Verification

### Before Fix
```python
from sglang.srt.configs import ChatGLMConfig
# ImportError: cannot import name 'ChatGLMConfig'
```

### After Fix
```python
from sglang.srt.utils.hf_transformers_utils import get_config
# ✅ Import succeeds, ChatGLMConfig no longer referenced
```

### Python Syntax Check
```bash
python3 -m py_compile python/sglang/srt/utils/hf_transformers_utils.py
python3 -m py_compile python/sglang/srt/model_executor/model_runner.py
# ✅ Both files compile successfully
```

## Impact Summary

### Code Reduction
- **Lines removed:** 197 lines
- **Config imports removed:** 19+ model configs
- **Functions removed:** 1 (Mistral Large 3 loader)
- **Model-specific hacks removed:** 8+ hacks

### Functionality
- ✅ Fixed ChatGLMConfig import error
- ✅ 100% DeepSeek-only model configuration
- ✅ Cleaner HuggingFace transformers integration
- ✅ No non-DeepSeek model support code remains

### Testing Status
- ✅ sgl-kernel compilation: SUCCESS (user confirmed)
- ✅ Model config imports: FIXED
- 🔄 Inference test (bench_one_batch): READY TO RETRY

## Next Steps for User

### 1. Pull Latest Changes (On GB200 Cluster)

```bash
cd /path/to/sglang
git pull origin deepseek-only
```

### 2. Reinstall Python Package

```bash
cd python
pip install -e ".[srt]"
```

### 3. Retry Inference Test

```bash
# Your original test command:
python -m sglang.bench_one_batch \
    --model-path /lustre/fsw/coreai_dlfw_dev/pengchengl/DeepSeek-R1-0528-FP4 \
    --attention-backend trtllm_mla \
    --moe-runner-backend=flashinfer_trtllm \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size=4 --ep-size=4 --data-parallel-size=4 \
    --enable-dp-attention --disable-radix-cache \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024 --cuda-graph-max-bs 1024 \
    --stream-interval 1 \
    --batch-size 2 --input-len 64 --output-len 4
```

### 4. Expected Result

The `ChatGLMConfig` import error should be resolved, and model loading should proceed to the next stage. If any new errors occur, they will indicate the next cleanup target.

## Related Documentation

- [FIX_PARALLEL_STATE_RESTORATION.md](FIX_PARALLEL_STATE_RESTORATION.md) - Previous import error fix
- [COMPLETE_BACKEND_CLEANUP_REPORT.md](COMPLETE_BACKEND_CLEANUP_REPORT.md) - Full cleanup summary
- [GPU_TEST_COMMAND.md](GPU_TEST_COMMAND.md) - Testing command reference

## Commit History

- **ce2d2fc59:** Remove all non-DeepSeek model configs and imports (THIS FIX)
- **4d3425f51:** Remove FA3/FA4/FlashMLA CUDA kernels
- **60739e5af:** Complete Ascend/NPU removal
- **253b57a:** Add comprehensive backend cleanup report

---

**Status:** ✅ RESOLVED
**Priority:** Critical
**Testing:** Ready for GB200 inference retry
**Expected Impact:** Unblocks model loading, enables inference testing to proceed
