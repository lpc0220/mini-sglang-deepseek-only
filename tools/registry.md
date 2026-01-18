# DeepSeek-Only Registry

This is the authoritative list of what's KEPT and REMOVED in the DeepSeek-only build.

---

## REMOVED

### Platform Backends
- **NPU/Ascend** - Huawei NPU support, MindSpore framework, NpuCommunicator
- **AMD/ROCm/HIP** - AMD GPU support, ROCm profiler, aiter library, HipCommunicator
- **Intel XPU** - Intel discrete GPU support, XpuCommunicator
- **Habana HPU** - Intel Gaudi accelerator, HpuCommunicator
- **CPU-only inference** - CPU-only mode (CPU quantization for weight loading kept)

### Attention Backends
- **Flash Attention 3/4** - FA3/FA4 kernels (sgl_kernel.flash_attn)
- **FlashMLA** - FlashMLA kernels (sgl_kernel.flash_mla), stubs remain in NSA
- **FlashAttention Backend** - flashattention_backend.py (entire file)
- **Flex Attention** - torch_flex_backend.py, TorchFlexAttnBackend (PyTorch flex_attention not used by DeepSeek)

### Models (~100+ removed)
- **Mamba/SSM** - State Space Models, MambaRadixCache, hybrid models
- **FLA** - Flash Linear Attention models
- **Vision/Multimodal** - deepseek_vl2.py, deepseek_janus_pro.py, deepseek_ocr.py
- **All other models** - LLaMA, Mistral, Qwen, GPT, Gemma, Phi, etc.

### Quantization
- **AWQ** - awq.py, awq_triton.py, awq_dequantize, awq_marlin_repack, awq_marlin_moe_repack (entire AWQ quantization removed including sgl-kernel functions)
- **GPTQ** - gptq.py, gptq_gemm, gptq_marlin_gemm, gptq_marlin_repack, gptq_shuffle, marlin_utils.py, marlin_utils_fp8.py, moe_wna16.py, auto_round.py, compressed_tensors/ (entire GPTQ/Marlin quantization removed)
- **Marlin MoE** - moe_runner/marlin.py, fused_marlin_moe.py, MoeRunnerBackend.MARLIN (Marlin MoE runner removed)
- **FP8 Marlin** - marlin_utils_fp8.py, apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin, can_auto_enable_marlin_fp8, SGLANG_FORCE_FP8_MARLIN (FP8 Marlin fallback removed - uses same GPTQ kernels)
- **GGUF** - gguf.py, GGUF weight loading

### Infrastructure
- **DLLM** - Diffusion LLM support, DllmConfig, is_dllm(), DLLM_EXTEND forward mode
- **gRPC** - grpc/ folder, grpc_server.py, grpc_mode
- **Transformers backend** - ModelImpl.TRANSFORMERS, TransformersForCausalLM fallback
- **MindSpore** - mindspore_runner.py, ModelImpl.MINDSPORE
- **encode_server** - Multimodal encoder disaggregation

### Other Removed
- CI/CD infrastructure (.github/, CI scripts)
- Docker infrastructure (Dockerfiles, K8s configs)
- Non-DeepSeek benchmarks
- Test infrastructure (unit tests, integration tests)

---

## KEPT

### Models (`sglang/python/sglang/srt/models/`)
- `deepseek.py` - DeepSeek v1
- `deepseek_v2.py` - DeepSeek v2/v3/R1
- `deepseek_nextn.py` - NextN speculative decoding
- `deepseek_common/` - Shared MLA, MoE components
- `registry.py`, `utils.py`

### Quantization (`layers/quantization/`)
- **FP8:** fp8.py, fp8_kernel.py, fp8_utils.py, fpgemm_fp8.py, w8a8_fp8.py
- **MXFP4:** mxfp4.py, mxfp4_tensor.py, kvfp4_tensor.py
- **INT8/W8A8:** int8_kernel.py, int8_utils.py, w8a8_int8.py, blockwise_int8.py
- **Other:** petit.py, petit_utils.py, qoq.py, modelopt_quant.py, w4afp8.py, quark/
- **Base:** base_config.py, kv_cache.py, unquant.py, utils.py

### Attention (`layers/attention/`)
- **FlashInfer:** flashinfer_backend.py, flashinfer_mla_backend.py
- **CutlassMLA:** cutlass_mla_backend.py
- **TRTLLm:** trtllm_mha_backend.py, trtllm_mla_backend.py
- **NSA:** nsa_backend.py, nsa/
- **Triton:** triton_backend.py, triton_ops/
- **Other:** tbo_backend.py, attention_registry.py, base_attn_backend.py, merge_state.py, utils.py

### MoE (`layers/moe/`)
- cutlass_moe.py, cutlass_moe_params.py, cutlass_w4a8_moe.py
- flashinfer_cutedsl_moe.py, fused_moe_native.py
- fused_moe_triton/, ep_moe/, moe_runner/, token_dispatcher/
- router.py, topk.py, routed_experts_capturer.py, kt_ep_wrapper.py, utils.py

### Function Call / Tool Calling (`function_call/`)
- deepseekv3_detector.py, deepseekv31_detector.py, deepseekv32_detector.py
- base_format_detector.py, function_call_parser.py, core_types.py, json_array_parser.py, utils.py

### Memory Cache (`mem_cache/`)
- radix_cache.py, radix_cache_cpp.py, hiradix_cache.py, chunk_cache.py
- memory_pool.py, memory_pool_host.py, swa_memory_pool.py, swa_radix_cache.py
- allocator.py, evict_policy.py, flush_cache.py, hicache_storage.py
- sparsity/, storage/, cpp_radix_tree/

### Distributed (`distributed/`)
- parallel_state.py, communication_op.py, naive_distributed.py
- device_communicators/

### Managers (`managers/`)
- scheduler.py + all scheduler mixins (dp_attn, pp, metrics, profiler, etc.)
- tp_worker.py, schedule_batch.py, schedule_policy.py
- tokenizer_manager.py, detokenizer_manager.py
- io_struct.py, template_manager.py, session_controller.py
- data_parallel_controller.py, cache_controller.py, disagg_service.py

### Entrypoints (`entrypoints/`)
- engine.py, EngineBase.py, http_server.py, http_server_engine.py
- openai/, ollama/, tool.py, warmup.py, context.py, harmony_utils.py

### sgl-kernel (`sgl-kernel/python/sgl_kernel/`)
- allreduce.py, attention.py, cutlass_moe.py, elementwise.py
- expert_specialization.py, fused_moe.py, gemm.py, hadamard.py
- kvcacheio.py, marlin.py, memory.py, moe.py, sampling.py
- sparse_flash_attn.py, spatial.py, speculative.py, top_k.py
- quantization/ (except gguf.py)

### Other Layers
- linear.py, activation.py, layernorm.py, rotary_embedding.py
- logits_processor.py, sampler.py, pooler.py
- vocab_parallel_embedding.py, model_parallel.py, parameter.py
- dp_attention.py, radix_attention.py
- communicator.py, communicator_nsa_cp.py, flashinfer_comm_fusion.py
- deep_gemm_wrapper/, modelopt_utils.py, torchao_utils.py

### Configs (`configs/`)
- model_config.py, device_config.py, load_config.py
- modelopt_config.py, update_config.py
