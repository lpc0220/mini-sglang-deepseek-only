# DeepSeek Dependency Graph

**Last Updated:** 2026-01-11
**Status:** Initial setup - Round 1

## Overview
This document tracks the dependency graph for DeepSeek models in SGLang. It is updated iteratively as we analyze the codebase.

## Legend
- 🟢 **Green (Keep):** DeepSeek models and confirmed dependencies
- 🔵 **Blue (Keep):** Required infrastructure (API, server, runtime)
- 🟡 **Yellow (Uncertain):** Needs analysis or user confirmation
- 🔴 **Red (Remove):** Provably unused by DeepSeek models

## Important Note
**Tool Calling Functionality:** Must be preserved! Location TBD (needs investigation).

---

## Round 1: Initial Survey (2026-01-11)

### DeepSeek Models - TEXT-ONLY R1 (🟢 Keep)
**USER DECISION:** Keep text-only DeepSeek R1 models only

```
sglang/python/sglang/srt/models/
├── deepseek.py          ✅ Text-only base model
├── deepseek_v2.py       ✅ Text-only v2/R1 (PRIMARY for DeepSeek-R1)
├── deepseek_nextn.py    ✅ Text-only variant
└── deepseek_common/     ✅ Shared MLA, MoE components
```

### DeepSeek Models - MULTIMODAL (🔴 Remove)
```
sglang/python/sglang/srt/models/
├── deepseek_janus_pro.py  ❌ Multimodal variant (not needed)
├── deepseek_ocr.py         ❌ OCR variant (not needed)
└── deepseek_vl2.py         ❌ Vision-Language variant (not needed)
```

### Status: Ready to begin removals

---

## Dependency Graph (Mermaid)

### High-Level Overview
```mermaid
graph TD
    subgraph "DeepSeek Models (Keep)"
        DS_V2[deepseek_v2.py]
        DS_COMMON[deepseek_common/]
        DS_VL2[deepseek_vl2.py]
        DS_JANUS[deepseek_janus_pro.py]
    end

    subgraph "Infrastructure (Keep)"
        API[API Endpoints]
        SERVER[Server Runtime]
        TOOLS[Tool Calling - TBD]
    end

    subgraph "To Be Analyzed"
        LAYERS[Layers/]
        KERNELS[Kernels/]
        MANAGERS[Managers/]
    end

    DS_V2 --> DS_COMMON
    DS_V2 --> LAYERS
    DS_V2 --> KERNELS

    SERVER --> DS_V2
    API --> SERVER

    style DS_V2 fill:#90EE90
    style DS_COMMON fill:#90EE90
    style API fill:#87CEEB
    style SERVER fill:#87CEEB
    style TOOLS fill:#FFFF99
```

---

## Next Steps
1. Survey all model files in `sglang/python/sglang/srt/models/`
2. Identify non-DeepSeek models (safe to remove)
3. Locate tool calling functionality
4. Create initial keep/remove lists
