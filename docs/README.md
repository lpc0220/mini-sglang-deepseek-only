# SGLang DeepSeek-Only Project Documentation

This directory contains all documentation for the SGLang shrinking project.

## Directory Structure

### `/project_tracking/`
Current project status and tracking files:
- [STATUS.md](project_tracking/STATUS.md) - Overall project status
- [QUICK_STATUS.md](project_tracking/QUICK_STATUS.md) - Quick status summary
- [PROGRESS_SUMMARY.md](project_tracking/PROGRESS_SUMMARY.md) - Detailed progress tracking
- [DEPENDENCIES.md](project_tracking/DEPENDENCIES.md) - DeepSeek dependency graph
- [REMOVED_FILES.md](project_tracking/REMOVED_FILES.md) - Log of all removed code

### `/phase_reports/`
Detailed reports from each project phase:
- [PHASE3_PLAN.md](phase_reports/PHASE3_PLAN.md) - Phase 3 implementation plan
- [PHASE3B_SUMMARY.md](phase_reports/PHASE3B_SUMMARY.md) - Phase 3B completion report
- [PHASE3C_*.md](phase_reports/) - Phase 3C detailed reports (multiple files)

## Quick Navigation

- **Main project plan:** [../CLAUDE.md](../CLAUDE.md)
- **Current status:** [project_tracking/STATUS.md](project_tracking/STATUS.md)
- **Latest phase report:** [phase_reports/](phase_reports/)

## Project Overview

This is a shrinking effort to reduce the SGLang codebase from 663K+ lines to a minimal version that:
- ONLY supports DeepSeek models (v2, v3, R1)
- Targets NVIDIA GPU only (removed CPU, NPU, AMD GPU support)
- Maintains 100% correctness with no functionality changes
- Keeps ALL quantization optimizations for NVIDIA

## Progress Summary

- **Original codebase:** ~663K lines (1945 Python files)
- **Current size:** ~296K lines (647 Python files)
- **Reduction:** ~367K lines removed (55.4%)
- **Current Phase:** Phase 3D - Infrastructure Cleanup (Complete)
- **Next Phase:** Build system updates and further cleanup
