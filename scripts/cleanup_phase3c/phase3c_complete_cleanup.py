#!/usr/bin/env python3
"""
Phase 3C: Complete NPU and HIP Conditional Removal

This script removes ALL remaining NPU and HIP conditional branches from SGLang,
achieving 100% NVIDIA CUDA-only codebase.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

class PlatformConditionalCleaner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.stats = {
            'files_processed': 0,
            'npu_branches_removed': 0,
            'hip_branches_removed': 0,
            'lines_removed': 0,
            'files_modified': [],
            'errors': []
        }

    def clean_file(self, file_path: Path) -> bool:
        """Clean NPU and HIP conditionals from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            original_lines = len(original_content.split('\n'))
            modified_content = original_content
            file_modified = False

            # Track changes
            npu_count = 0
            hip_count = 0

            # Pattern 1: Remove standalone NPU blocks
            # if _is_npu:\n    ...\nelse:\n    ...
            pattern1 = re.compile(
                r'(\s*)if _is_npu:\n((?:.*\n)*?)\1else:\n((?:.*\n)*?)(?=\1(?:if|elif|else|def|class|@|\Z|[^\s]))',
                re.MULTILINE
            )
            new_content = pattern1.sub(r'\1\3', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern1.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 2: Remove elif _is_npu branches
            # elif _is_npu:\n    ...
            pattern2 = re.compile(
                r'(\s*)elif _is_npu:\n((?:(?!\1(?:elif|else|def|class|@)).*\n)*)',
                re.MULTILINE
            )
            new_content = pattern2.sub('', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern2.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 3: Remove standalone HIP blocks
            pattern3 = re.compile(
                r'(\s*)if _is_hip:\n((?:.*\n)*?)\1else:\n((?:.*\n)*?)(?=\1(?:if|elif|else|def|class|@|\Z|[^\s]))',
                re.MULTILINE
            )
            new_content = pattern3.sub(r'\1\3', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern3.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 4: Remove elif _is_hip branches
            pattern4 = re.compile(
                r'(\s*)elif _is_hip:\n((?:(?!\1(?:elif|else|def|class|@)).*\n)*)',
                re.MULTILINE
            )
            new_content = pattern4.sub('', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern4.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 5: Simplify OR conditions with NPU
            # if _is_cuda or _is_npu: -> if _is_cuda:
            pattern5 = re.compile(r'if (_is_cuda) or _is_npu:')
            new_content = pattern5.sub(r'if \1:', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern5.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 6: Simplify OR conditions with HIP
            # if _is_cuda or _is_hip: -> if _is_cuda:
            pattern6 = re.compile(r'if (_is_cuda) or _is_hip:')
            new_content = pattern6.sub(r'if \1:', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern6.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 7: Remove NPU variable assignments
            pattern7 = re.compile(r'^\s*_is_npu\s*=.*$\n?', re.MULTILINE)
            new_content = pattern7.sub('', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern7.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 8: Remove HIP variable assignments
            pattern8 = re.compile(r'^\s*_is_hip\s*=.*$\n?', re.MULTILINE)
            new_content = pattern8.sub('', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern8.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 9: Remove _use_aiter variable (HIP-specific)
            pattern9 = re.compile(r'^\s*_use_aiter\s*=.*$\n?', re.MULTILINE)
            new_content = pattern9.sub('', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern9.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 10: Remove NPU imports
            pattern10 = re.compile(r'^\s*from.*npu.*import.*$\n?', re.MULTILINE)
            new_content = pattern10.sub('', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern10.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 11: Remove remaining if _is_npu: blocks without else
            pattern11 = re.compile(
                r'(\s*)if _is_npu:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)',
                re.MULTILINE
            )
            new_content = pattern11.sub('', modified_content)
            if new_content != modified_content:
                npu_count += len(pattern11.findall(modified_content))
                modified_content = new_content
                file_modified = True

            # Pattern 12: Remove remaining if _is_hip: blocks without else
            pattern12 = re.compile(
                r'(\s*)if _is_hip:\n((?:(?!\1(?:if|elif|else|def|class|@)).*\n)*)',
                re.MULTILINE
            )
            new_content = pattern12.sub('', modified_content)
            if new_content != modified_content:
                hip_count += len(pattern12.findall(modified_content))
                modified_content = new_content
                file_modified = True

            if file_modified:
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

                modified_lines = len(modified_content.split('\n'))
                lines_removed = original_lines - modified_lines

                self.stats['files_modified'].append(str(file_path))
                self.stats['npu_branches_removed'] += npu_count
                self.stats['hip_branches_removed'] += hip_count
                self.stats['lines_removed'] += lines_removed

                print(f"✓ {file_path.name}: removed {npu_count} NPU + {hip_count} HIP branches ({lines_removed} lines)")

            self.stats['files_processed'] += 1
            return True

        except Exception as e:
            self.stats['errors'].append(f"{file_path}: {str(e)}")
            print(f"✗ {file_path.name}: ERROR - {str(e)}")
            return False

    def find_files_with_platform_refs(self) -> List[Path]:
        """Find all Python files with NPU or HIP references."""
        files = []
        for py_file in self.root_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '_is_npu' in content or '_is_hip' in content:
                        files.append(py_file)
            except Exception as e:
                print(f"Warning: Could not read {py_file}: {e}")
        return files

    def run(self):
        """Execute the complete cleanup."""
        print("=" * 80)
        print("Phase 3C: Complete NPU and HIP Conditional Removal")
        print("=" * 80)
        print()

        # Find all files
        print("Scanning for files with NPU/HIP references...")
        files = self.find_files_with_platform_refs()
        print(f"Found {len(files)} files to process")
        print()

        # Process each file
        print("Processing files...")
        print("-" * 80)
        for file_path in sorted(files):
            self.clean_file(file_path)

        print("-" * 80)
        print()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print cleanup summary."""
        print("=" * 80)
        print("CLEANUP SUMMARY")
        print("=" * 80)
        print(f"Files processed:        {self.stats['files_processed']}")
        print(f"Files modified:         {len(self.stats['files_modified'])}")
        print(f"NPU branches removed:   {self.stats['npu_branches_removed']}")
        print(f"HIP branches removed:   {self.stats['hip_branches_removed']}")
        print(f"Total lines removed:    {self.stats['lines_removed']}")
        print(f"Errors encountered:     {len(self.stats['errors'])}")
        print()

        if self.stats['errors']:
            print("ERRORS:")
            for error in self.stats['errors']:
                print(f"  - {error}")
            print()

        print("Modified files:")
        for file_path in self.stats['files_modified'][:20]:  # Show first 20
            print(f"  - {file_path}")
        if len(self.stats['files_modified']) > 20:
            print(f"  ... and {len(self.stats['files_modified']) - 20} more")
        print()

        # Verify remaining references
        print("Verifying remaining platform references...")
        remaining = self.count_remaining_refs()
        print(f"Remaining _is_npu refs: {remaining['npu']}")
        print(f"Remaining _is_hip refs: {remaining['hip']}")
        print()

        if remaining['npu'] == 0 and remaining['hip'] == 0:
            print("✅ SUCCESS: All NPU and HIP conditionals removed!")
        else:
            print("⚠️  WARNING: Some platform references remain (may be in comments/strings)")
        print("=" * 80)

    def count_remaining_refs(self) -> Dict[str, int]:
        """Count remaining NPU and HIP references."""
        npu_count = 0
        hip_count = 0

        for py_file in self.root_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    npu_count += content.count('_is_npu')
                    hip_count += content.count('_is_hip')
            except:
                pass

        return {'npu': npu_count, 'hip': hip_count}


def main():
    # Target sglang/python/sglang/srt directory
    root_dir = '/Users/lpc/workspace/sglang-deepseek-only/sglang/python/sglang/srt'

    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return

    cleaner = PlatformConditionalCleaner(root_dir)
    cleaner.run()


if __name__ == '__main__':
    main()
