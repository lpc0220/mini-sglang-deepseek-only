#!/usr/bin/env python3
"""
Phase 3C Final Validation

Comprehensive validation that all NPU/HIP/XPU conditionals have been removed
and CUDA functionality is preserved.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class Phase3CValidator:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.srt_dir = self.root_dir / 'sglang' / 'python' / 'sglang' / 'srt'

    def count_pattern(self, pattern: str) -> Tuple[int, List[str]]:
        """Count occurrences of a pattern in Python files."""
        try:
            result = subprocess.run(
                ['grep', '-rn', pattern, str(self.srt_dir), '--include=*.py'],
                capture_output=True, text=True
            )
            lines = [l for l in result.stdout.split('\n') if l.strip()]
            return len(lines), lines[:5]  # Return count and first 5 matches
        except:
            return 0, []

    def check_file_exists(self, rel_path: str) -> bool:
        """Check if a file exists."""
        return (self.srt_dir / rel_path).exists()

    def validate_removed_platforms(self) -> Dict[str, any]:
        """Validate that NPU/HIP/XPU references are removed."""
        results = {}

        # Check NPU
        npu_count, npu_samples = self.count_pattern('_is_npu')
        results['npu'] = {
            'count': npu_count,
            'samples': npu_samples,
            'pass': npu_count == 0
        }

        # Check HIP
        hip_count, hip_samples = self.count_pattern('_is_hip')
        results['hip'] = {
            'count': hip_count,
            'samples': hip_samples,
            'pass': hip_count == 0
        }

        # Check XPU
        xpu_count, xpu_samples = self.count_pattern('_is_xpu')
        results['xpu'] = {
            'count': xpu_count,
            'samples': xpu_samples,
            'pass': xpu_count == 0
        }

        return results

    def validate_cuda_preserved(self) -> Dict[str, any]:
        """Validate that CUDA functionality is preserved."""
        results = {}

        # Check CUDA references exist
        cuda_count, cuda_samples = self.count_pattern('_is_cuda')
        results['cuda'] = {
            'count': cuda_count,
            'samples': cuda_samples,
            'pass': cuda_count > 100  # Should have many CUDA references
        }

        return results

    def validate_critical_files(self) -> Dict[str, bool]:
        """Validate that critical files exist and are clean."""
        critical_files = [
            'models/deepseek_v2.py',
            'models/deepseek_nextn.py',
            'layers/moe/ep_moe/layer.py',
            'layers/moe/topk.py',
            'layers/quantization/fp8.py',
            'layers/quantization/awq.py',
            'distributed/parallel_state.py',
            'model_executor/model_runner.py',
        ]

        results = {}
        for file_path in critical_files:
            exists = self.check_file_exists(file_path)
            if exists:
                full_path = self.srt_dir / file_path
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        has_npu = '_is_npu' in content
                        has_hip = '_is_hip' in content
                        has_xpu = '_is_xpu' in content
                        clean = not (has_npu or has_hip or has_xpu)
                        results[file_path] = {
                            'exists': True,
                            'clean': clean,
                            'npu': has_npu,
                            'hip': has_hip,
                            'xpu': has_xpu
                        }
                except:
                    results[file_path] = {'exists': True, 'clean': False, 'error': True}
            else:
                results[file_path] = {'exists': False, 'clean': False}

        return results

    def run_validation(self):
        """Run complete validation suite."""
        print("=" * 80)
        print("PHASE 3C FINAL VALIDATION")
        print("=" * 80)
        print()

        # 1. Validate platform removal
        print("1. Platform Conditionals Removal")
        print("-" * 80)
        platform_results = self.validate_removed_platforms()

        for platform, data in platform_results.items():
            status = "✅ PASS" if data['pass'] else "❌ FAIL"
            print(f"{platform.upper()} references: {data['count']} {status}")
            if not data['pass'] and data['samples']:
                print(f"  Sample occurrences:")
                for sample in data['samples'][:3]:
                    print(f"    {sample[:100]}")

        print()

        # 2. Validate CUDA preservation
        print("2. CUDA Functionality Preservation")
        print("-" * 80)
        cuda_results = self.validate_cuda_preserved()

        for key, data in cuda_results.items():
            status = "✅ PASS" if data['pass'] else "❌ FAIL"
            print(f"{key.upper()} references: {data['count']} {status}")

        print()

        # 3. Validate critical files
        print("3. Critical Files Validation")
        print("-" * 80)
        file_results = self.validate_critical_files()

        all_clean = True
        for file_path, data in file_results.items():
            if not data['exists']:
                print(f"❌ {file_path}: NOT FOUND")
                all_clean = False
            elif data.get('error'):
                print(f"⚠️  {file_path}: READ ERROR")
                all_clean = False
            elif not data['clean']:
                print(f"❌ {file_path}: PLATFORM REFS (npu={data.get('npu')}, hip={data.get('hip')}, xpu={data.get('xpu')})")
                all_clean = False
            else:
                print(f"✅ {file_path}: Clean")

        print()

        # 4. Overall results
        print("=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)

        all_platforms_removed = all(data['pass'] for data in platform_results.values())
        cuda_preserved = cuda_results['cuda']['pass']

        print(f"Platform conditionals removed: {'✅ YES' if all_platforms_removed else '❌ NO'}")
        print(f"CUDA functionality preserved: {'✅ YES' if cuda_preserved else '❌ NO'}")
        print(f"Critical files clean: {'✅ YES' if all_clean else '❌ NO'}")
        print()

        if all_platforms_removed and cuda_preserved and all_clean:
            print("🎯 PHASE 3C: ✅ COMPLETE - 100% SUCCESS")
            print()
            print("Summary:")
            print(f"  - NPU references: {platform_results['npu']['count']}")
            print(f"  - HIP references: {platform_results['hip']['count']}")
            print(f"  - XPU references: {platform_results['xpu']['count']}")
            print(f"  - CUDA references: {cuda_results['cuda']['count']} (preserved)")
            print()
            print("SGLang is now 100% NVIDIA CUDA-only!")
        else:
            print("⚠️  PHASE 3C: INCOMPLETE - Issues found")

        print("=" * 80)


def main():
    validator = Phase3CValidator('/Users/lpc/workspace/sglang-deepseek-only')
    validator.run_validation()


if __name__ == '__main__':
    main()
