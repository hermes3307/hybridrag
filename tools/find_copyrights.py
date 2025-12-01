#!/usr/bin/env python3
"""
Copyright Header Finder with Product Analytics
Scans .c, .java, and .h files for copyright headers and categorizes by product/project
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# File extensions to search
EXTENSIONS = ['.c', '.java', '.h']

# Number of lines to check from the start of each file
HEADER_LINES = 30

# Product/Project names to categorize
PRODUCTS = [
    'Altibase',
    'VADO',
    'QCubic',
    'Cisphere',
    'ShardSphere',
    'ShardShere',  # Possible typo variant
]

# Open Source License patterns to detect
OPEN_SOURCE_LICENSES = {
    'GNU GPL': ['GNU General Public License', 'GNU GPL', 'GPL-2', 'GPL-3', 'GPLv2', 'GPLv3'],
    'GNU LGPL': ['GNU Lesser General Public License', 'GNU LGPL', 'LGPL-2', 'LGPL-3', 'LGPLv2', 'LGPLv3'],
    'BSD': ['BSD License', 'BSD-2-Clause', 'BSD-3-Clause', 'BSD 2-Clause', 'BSD 3-Clause', 'Berkeley Software Distribution'],
    'Apache': ['Apache License', 'Apache-2.0', 'Apache 2.0', 'Apache Software License'],
    'MIT': ['MIT License', 'MIT license'],
    'Mozilla': ['Mozilla Public License', 'MPL-1', 'MPL-2', 'MPLv1', 'MPLv2'],
    'Free Software': ['Free Software Foundation', 'FSF', 'free software'],
    'Open Source': ['open source', 'opensource'],
    'Creative Commons': ['Creative Commons', 'CC BY', 'CC-BY'],
    'Eclipse': ['Eclipse Public License', 'EPL'],
}

def find_files(root_dir, extensions):
    """Recursively find all files with specified extensions"""
    files = []
    root_path = Path(root_dir)

    for ext in extensions:
        files.extend(root_path.rglob(f'*{ext}'))

    return sorted(files)

def detect_products(text):
    """Detect which products are mentioned in the text"""
    found_products = []
    text_lower = text.lower()

    for product in PRODUCTS:
        if product.lower() in text_lower:
            found_products.append(product)

    return found_products

def detect_licenses(text):
    """Detect which open source licenses are mentioned in the text"""
    found_licenses = []
    text_lower = text.lower()

    for license_name, patterns in OPEN_SOURCE_LICENSES.items():
        for pattern in patterns:
            if pattern.lower() in text_lower:
                if license_name not in found_licenses:
                    found_licenses.append(license_name)
                break

    return found_licenses

def extract_copyright(file_path, num_lines=HEADER_LINES):
    """Extract copyright information from file header"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(num_lines)]
            header = ''.join(lines)

        # Check if copyright exists (case-insensitive)
        copyright_pattern = re.compile(r'copyright|©|\(c\)', re.IGNORECASE)

        has_copyright = bool(copyright_pattern.search(header))

        # Find copyright block boundaries
        copyright_lines = []
        in_copyright_block = False
        empty_line_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if line contains copyright-related keywords
            if copyright_pattern.search(line):
                in_copyright_block = True
                empty_line_count = 0

            if in_copyright_block:
                # Check for end of comment block
                if stripped and (
                    stripped.startswith('#include') or
                    stripped.startswith('package ') or
                    stripped.startswith('import ') or
                    stripped.startswith('public class') or
                    stripped.startswith('class ') or
                    stripped.startswith('void ') or
                    stripped.startswith('int ') or
                    stripped.startswith('#ifndef') or
                    stripped.startswith('#define')
                ):
                    # We've reached actual code, stop here
                    break

                copyright_lines.append(line.rstrip())

                # Track consecutive empty lines
                if not stripped:
                    empty_line_count += 1
                    # If we have 2+ consecutive empty lines, likely end of header
                    if empty_line_count >= 2:
                        break
                else:
                    empty_line_count = 0

                # Stop if we've collected enough context
                if len(copyright_lines) > 20:
                    break

        # Clean up the copyright block
        copyright_text = None
        if copyright_lines:
            # Remove leading/trailing empty lines
            while copyright_lines and not copyright_lines[0].strip():
                copyright_lines.pop(0)
            while copyright_lines and not copyright_lines[-1].strip():
                copyright_lines.pop()

            copyright_text = '\n'.join(copyright_lines) if copyright_lines else None

        # Detect products in the entire header
        products = detect_products(header)

        # Detect licenses in the entire header
        licenses = detect_licenses(header)
        is_open_source = len(licenses) > 0

        return {
            'has_copyright': has_copyright,
            'copyright_text': copyright_text,
            'products': products,
            'licenses': licenses,
            'is_open_source': is_open_source,
            'header': header
        }

    except Exception as e:
        return {
            'has_copyright': False,
            'copyright_text': None,
            'products': [],
            'licenses': [],
            'is_open_source': False,
            'header': None,
            'error': str(e)
        }

def generate_report(root_dir, output_file='report.txt'):
    """Generate copyright report with analytics"""
    files = find_files(root_dir, EXTENSIONS)

    if not files:
        print(f"No files with extensions {EXTENSIONS} found in {root_dir}")
        return

    # Data structures for categorization
    files_with_copyright = []
    files_without_copyright = []
    copyright_groups = defaultdict(list)
    product_files = defaultdict(list)
    files_by_extension = defaultdict(int)
    product_stats = defaultdict(lambda: {'with_copyright': 0, 'without_copyright': 0, 'total': 0, 'open_source': 0, 'proprietary': 0})
    unclassified_files = []

    # License categorization
    open_source_files = []
    proprietary_files = []
    license_stats = defaultdict(int)
    license_files = defaultdict(list)

    print(f"Scanning {len(files)} files for copyright headers...")

    for file_path in files:
        info = extract_copyright(file_path)
        file_ext = file_path.suffix

        # Count by extension
        files_by_extension[file_ext] += 1

        # Categorize by copyright presence
        if info['has_copyright'] and info['copyright_text']:
            files_with_copyright.append((file_path, info))
            copyright_groups[info['copyright_text']].append(file_path)
        else:
            files_without_copyright.append((file_path, info))

        # Categorize by license type
        if info['has_copyright']:
            if info['is_open_source']:
                open_source_files.append((file_path, info))
                for license_name in info['licenses']:
                    license_stats[license_name] += 1
                    license_files[license_name].append((file_path, info))
            else:
                proprietary_files.append((file_path, info))

        # Categorize by product
        if info['products']:
            for product in info['products']:
                product_files[product].append((file_path, info))
                product_stats[product]['total'] += 1
                if info['has_copyright']:
                    product_stats[product]['with_copyright'] += 1
                else:
                    product_stats[product]['without_copyright'] += 1

                # Track open source vs proprietary by product
                if info['is_open_source']:
                    product_stats[product]['open_source'] += 1
                elif info['has_copyright']:
                    product_stats[product]['proprietary'] += 1
        else:
            unclassified_files.append((file_path, info))

    # Generate report
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("COPYRIGHT HEADER ANALYSIS REPORT")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Root Directory: {root_dir}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # SECTION 1: EXECUTIVE SUMMARY
    output_lines.append("EXECUTIVE SUMMARY")
    output_lines.append("-" * 80)
    output_lines.append(f"Total Files Scanned:        {len(files)}")
    output_lines.append(f"Files WITH Copyright:       {len(files_with_copyright)} ({len(files_with_copyright)*100//len(files) if files else 0}%)")
    output_lines.append(f"Files WITHOUT Copyright:    {len(files_without_copyright)} ({len(files_without_copyright)*100//len(files) if files else 0}%)")
    output_lines.append("")
    output_lines.append(f"Open Source Licensed:       {len(open_source_files)} ({len(open_source_files)*100//len(files) if files else 0}%)")
    output_lines.append(f"Proprietary Licensed:       {len(proprietary_files)} ({len(proprietary_files)*100//len(files) if files else 0}%)")
    output_lines.append("")
    output_lines.append(f"Unique Copyright Types:     {len(copyright_groups)}")
    output_lines.append(f"Products Detected:          {len(product_stats)}")
    output_lines.append(f"Unclassified Files:         {len(unclassified_files)}")
    output_lines.append("")

    # File type breakdown
    output_lines.append("File Type Breakdown:")
    for ext, count in sorted(files_by_extension.items()):
        output_lines.append(f"  {ext:8s} : {count:5d} files")
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("")

    # SECTION 2: LICENSE ANALYTICS
    if license_stats:
        output_lines.append("OPEN SOURCE LICENSE ANALYTICS")
        output_lines.append("-" * 80)
        output_lines.append(f"Total Open Source Files:    {len(open_source_files)}")
        output_lines.append("")

        output_lines.append("License Distribution:")
        for license_name in sorted(license_stats.keys(), key=lambda x: license_stats[x], reverse=True):
            count = license_stats[license_name]
            output_lines.append(f"  {license_name:20s} : {count:5d} files ({count*100//len(files) if files else 0}%)")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

    # SECTION 3: PRODUCT ANALYTICS
    if product_stats:
        output_lines.append("PRODUCT/PROJECT ANALYTICS")
        output_lines.append("-" * 80)
        output_lines.append("")

        for product in sorted(product_stats.keys()):
            stats = product_stats[product]
            output_lines.append(f"Product: {product}")
            output_lines.append(f"  Total Files:             {stats['total']}")
            output_lines.append(f"  With Copyright:          {stats['with_copyright']} ({stats['with_copyright']*100//stats['total'] if stats['total'] else 0}%)")
            output_lines.append(f"  Without Copyright:       {stats['without_copyright']} ({stats['without_copyright']*100//stats['total'] if stats['total'] else 0}%)")
            output_lines.append(f"  Open Source:             {stats['open_source']} ({stats['open_source']*100//stats['total'] if stats['total'] else 0}%)")
            output_lines.append(f"  Proprietary:             {stats['proprietary']} ({stats['proprietary']*100//stats['total'] if stats['total'] else 0}%)")
            output_lines.append("")

        output_lines.append("=" * 80)
        output_lines.append("")

    # SECTION 4: OPEN SOURCE FILES BY LICENSE
    if license_files:
        output_lines.append("OPEN SOURCE FILES BY LICENSE TYPE")
        output_lines.append("-" * 80)
        output_lines.append("")

        for license_name in sorted(license_files.keys()):
            file_list = license_files[license_name]
            output_lines.append(f"License: {license_name} ({len(file_list)} files)")
            output_lines.append("-" * 80)

            for file_path, info in file_list:
                output_lines.append(f"\n{file_path}")

                # Show which products this file belongs to
                if info['products']:
                    output_lines.append(f"  Products: {', '.join(info['products'])}")

                if info['copyright_text']:
                    output_lines.append("  Copyright Header:")
                    for line in info['copyright_text'].split('\n')[:10]:  # First 10 lines
                        output_lines.append(f"    {line}")
                    if len(info['copyright_text'].split('\n')) > 10:
                        output_lines.append("    ...")

            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")

    # SECTION 5: PROPRIETARY FILES
    if proprietary_files:
        output_lines.append("PROPRIETARY/NON-OPEN-SOURCE FILES")
        output_lines.append("-" * 80)
        output_lines.append(f"Total: {len(proprietary_files)} files")
        output_lines.append("")

        # Group by product
        proprietary_by_product = defaultdict(list)
        proprietary_unclassified = []

        for file_path, info in proprietary_files:
            if info['products']:
                for product in info['products']:
                    proprietary_by_product[product].append((file_path, info))
            else:
                proprietary_unclassified.append((file_path, info))

        if proprietary_by_product:
            for product in sorted(proprietary_by_product.keys()):
                files_list = proprietary_by_product[product]
                output_lines.append(f"\n{product}: {len(files_list)} files")
                output_lines.append("-" * 40)
                for fp, info in files_list[:10]:  # Show first 10
                    output_lines.append(f"  {fp}")
                if len(files_list) > 10:
                    output_lines.append(f"  ... and {len(files_list) - 10} more")

        if proprietary_unclassified:
            output_lines.append(f"\nUnclassified Proprietary: {len(proprietary_unclassified)} files")
            output_lines.append("-" * 40)
            for fp, info in proprietary_unclassified[:10]:
                output_lines.append(f"  {fp}")
            if len(proprietary_unclassified) > 10:
                output_lines.append(f"  ... and {len(proprietary_unclassified) - 10} more")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

    # SECTION 6: DETAILED PRODUCT FILE LISTINGS
    if product_files:
        output_lines.append("DETAILED PRODUCT FILE LISTINGS")
        output_lines.append("-" * 80)
        output_lines.append("")

        for product in sorted(product_files.keys()):
            file_list = product_files[product]
            output_lines.append(f"Product: {product} ({len(file_list)} files)")
            output_lines.append("-" * 80)

            for file_path, info in file_list:
                copyright_status = "WITH COPYRIGHT" if info['has_copyright'] else "NO COPYRIGHT"
                license_type = "OPEN SOURCE" if info['is_open_source'] else "PROPRIETARY"

                status_str = f"{copyright_status} | {license_type}"
                if info['licenses']:
                    status_str += f" | {', '.join(info['licenses'])}"

                output_lines.append(f"\n[{status_str}] {file_path}")

                if info['copyright_text']:
                    output_lines.append("Copyright Header:")
                    for line in info['copyright_text'].split('\n'):
                        output_lines.append(f"  {line}")
                elif info['has_copyright']:
                    output_lines.append("  (Copyright detected but header extraction failed)")

            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")

    # SECTION 7: UNCLASSIFIED FILES
    if unclassified_files:
        output_lines.append("UNCLASSIFIED FILES (No Product Detected)")
        output_lines.append("-" * 80)
        output_lines.append(f"Total: {len(unclassified_files)} files")
        output_lines.append("")

        for file_path, info in unclassified_files:
            copyright_status = "WITH COPYRIGHT" if info['has_copyright'] else "NO COPYRIGHT"
            output_lines.append(f"\n[{copyright_status}] {file_path}")

            if info['copyright_text']:
                output_lines.append("Copyright Header:")
                for line in info['copyright_text'].split('\n'):
                    output_lines.append(f"  {line}")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

    # SECTION 8: FILES WITHOUT COPYRIGHT
    if files_without_copyright:
        output_lines.append("FILES WITHOUT COPYRIGHT HEADERS (FULL LIST)")
        output_lines.append("-" * 80)
        output_lines.append(f"Total: {len(files_without_copyright)} files")
        output_lines.append("")

        # Group by product if detected
        without_copyright_by_product = defaultdict(list)
        without_copyright_unclassified = []

        for file_path, info in files_without_copyright:
            if info['products']:
                for product in info['products']:
                    without_copyright_by_product[product].append(file_path)
            else:
                without_copyright_unclassified.append(file_path)

        if without_copyright_by_product:
            for product in sorted(without_copyright_by_product.keys()):
                files_list = without_copyright_by_product[product]
                output_lines.append(f"\n{product}: {len(files_list)} files")
                for fp in files_list:
                    output_lines.append(f"  {fp}")

        if without_copyright_unclassified:
            output_lines.append(f"\nUnclassified: {len(without_copyright_unclassified)} files")
            for fp in without_copyright_unclassified:
                output_lines.append(f"  {fp}")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append("")

    # SECTION 9: UNIQUE COPYRIGHT TYPES
    if len(copyright_groups) > 0:
        output_lines.append("UNIQUE COPYRIGHT HEADERS FOUND")
        output_lines.append("-" * 80)
        output_lines.append(f"Total unique copyright types: {len(copyright_groups)}")
        output_lines.append("")

        for i, (copyright_text, file_list) in enumerate(sorted(copyright_groups.items(), key=lambda x: -len(x[1])), 1):
            output_lines.append(f"\nCopyright Type #{i} ({len(file_list)} files)")
            output_lines.append("-" * 40)
            output_lines.append(copyright_text)
            output_lines.append("")
            output_lines.append("Files using this copyright:")
            for fp in file_list[:15]:  # Show first 15 files
                output_lines.append(f"  - {fp}")
            if len(file_list) > 15:
                output_lines.append(f"  ... and {len(file_list) - 15} more")
            output_lines.append("")

        output_lines.append("=" * 80)
        output_lines.append("")

    # Write output
    report_text = '\n'.join(output_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{'=' * 80}")
    print(f"Report generated successfully!")
    print(f"Output file: {output_file}")
    print(f"{'=' * 80}")
    print(f"\nQuick Summary:")
    print(f"  Total files scanned: {len(files)}")
    print(f"  With copyright: {len(files_with_copyright)}")
    print(f"  Without copyright: {len(files_without_copyright)}")
    print(f"  Open source: {len(open_source_files)}")
    print(f"  Proprietary: {len(proprietary_files)}")
    print(f"  Products detected: {len(product_stats)}")
    if product_stats:
        print(f"\n  Products found:")
        for product in sorted(product_stats.keys()):
            stats = product_stats[product]
            print(f"    - {product}: {stats['total']} files ({stats['open_source']} open source, {stats['proprietary']} proprietary)")
    if license_stats:
        print(f"\n  Open Source Licenses found:")
        for license_name in sorted(license_stats.keys()):
            print(f"    - {license_name}: {license_stats[license_name]} files")
    print(f"{'=' * 80}")

    return report_text

def main():
    """Main function"""
    # Default to current directory if no arguments provided
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'report.txt'

    # Show usage if help is requested
    if root_dir in ['-h', '--help', 'help']:
        print("Copyright Header Finder with Product Analytics")
        print("\nUsage: python3 find_copyrights.py [directory] [output_file]")
        print("\nArguments:")
        print("  directory    : Directory to scan (default: current directory)")
        print("  output_file  : Output report file (default: report.txt)")
        print("\nExamples:")
        print("  python3 find_copyrights.py")
        print("  python3 find_copyrights.py /path/to/project")
        print("  python3 find_copyrights.py /path/to/project my_report.txt")
        print("  python3 find_copyrights.py . custom_analysis.txt")
        sys.exit(0)

    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory")
        sys.exit(1)

    # Convert to absolute path for better display
    root_dir = os.path.abspath(root_dir)

    generate_report(root_dir, output_file)

if __name__ == '__main__':
    main()
