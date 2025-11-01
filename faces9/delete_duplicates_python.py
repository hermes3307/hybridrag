#!/usr/bin/env python3
"""
Delete All Duplicate Images - Python Script

Removes duplicate image files, keeping only the OLDEST copy of each unique image.
Also removes corresponding JSON metadata files.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json

# Configuration
FACES_DIR = Path('/home/pi/faces')
BACKUP_DIR = Path('./duplicate_backups')
DRY_RUN = '--confirm' not in sys.argv

def main():
    print("=" * 80)
    print("🗑️  DELETE ALL DUPLICATES - Python Version")
    print("=" * 80)
    print()

    # Check if directory exists
    if not FACES_DIR.exists():
        print(f"❌ Error: Directory not found: {FACES_DIR}")
        sys.exit(1)

    print(f"📁 Scanning directory: {FACES_DIR}")
    print()

    # Find all JPG files
    print("⏳ Finding all image files...")
    jpg_files = list(FACES_DIR.glob('face_*.jpg'))
    total_files = len(jpg_files)
    print(f"✅ Found {total_files:,} image files")
    print()

    # Group files by hash
    print("🔍 Grouping files by hash...")
    files_by_hash = defaultdict(list)

    for file_path in jpg_files:
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            hash_part = parts[-1]
            # Get modification time
            mtime = file_path.stat().st_mtime
            files_by_hash[hash_part].append((mtime, file_path))

    # Sort each group by modification time (oldest first)
    for hash_val in files_by_hash:
        files_by_hash[hash_val].sort(key=lambda x: x[0])

    # Find duplicates
    duplicates_to_delete = []
    files_to_keep = []

    for hash_val, file_list in files_by_hash.items():
        if len(file_list) > 1:
            # Keep the oldest (first in sorted list)
            files_to_keep.append(file_list[0][1])
            # Mark the rest for deletion
            for mtime, file_path in file_list[1:]:
                duplicates_to_delete.append(file_path)
        else:
            files_to_keep.append(file_list[0][1])

    unique_hashes = len(files_by_hash)
    duplicate_count = len(duplicates_to_delete)

    print(f"✅ Analysis complete")
    print()

    # Statistics
    print("=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Total Image Files:        {total_files:,}")
    print(f"Unique Images (by hash):  {unique_hashes:,}")
    print(f"Duplicate Files:          {duplicate_count:,}")
    print(f"Files to Keep:            {len(files_to_keep):,}")
    print(f"Files to Delete:          {duplicate_count:,}")
    print(f"Space to save:            ~{duplicate_count * 2:,} files (images + JSON)")
    print()

    if duplicate_count == 0:
        print("✅ No duplicates found! All images are unique.")
        return

    # Show sample duplicates
    if not DRY_RUN and duplicate_count > 10:
        print("=" * 80)
        print("🔍 SAMPLE DUPLICATES (showing first 10)")
        print("=" * 80)
        for i, dup_file in enumerate(duplicates_to_delete[:10], 1):
            print(f"{i}. {dup_file.name}")
        print(f"... and {duplicate_count - 10:,} more files")
        print()

    # Dry-run or actual deletion
    if DRY_RUN:
        print("=" * 80)
        print("⚠️  DRY-RUN MODE - NO FILES DELETED")
        print("=" * 80)
        print()
        print("This was a dry-run. No files were actually deleted.")
        print()
        print(f"Found {duplicate_count:,} duplicate files to delete.")
        print()
        print("To actually delete these files, run:")
        print("  python3 delete_duplicates_python.py --confirm")
        print()
    else:
        print("=" * 80)
        print("⚠️  DELETION MODE - FILES WILL BE DELETED")
        print("=" * 80)
        print()

        # Create backup directory
        BACKUP_DIR.mkdir(exist_ok=True)

        # Save backup list
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = BACKUP_DIR / f"deleted_files_{timestamp}.txt"

        print(f"💾 Creating backup list: {backup_file}")
        with open(backup_file, 'w') as f:
            for dup_file in duplicates_to_delete:
                f.write(f"{dup_file}\n")
        print(f"✅ Backup list saved")
        print()

        # Confirm deletion
        print(f"⚠️  About to delete {duplicate_count:,} duplicate image files and their JSON files.")
        print("⚠️  This action CANNOT be undone!")
        print()
        response = input("Type 'DELETE' to confirm: ").strip()

        if response != 'DELETE':
            print("❌ Deletion cancelled.")
            return

        print()
        print("🗑️  Deleting files...")
        print()

        deleted_count = 0
        error_count = 0

        # Progress indicator
        total_to_delete = duplicate_count
        for idx, dup_file in enumerate(duplicates_to_delete, 1):
            try:
                # Delete image file
                if dup_file.exists():
                    dup_file.unlink()
                    deleted_count += 1

                # Delete corresponding JSON file
                json_file = dup_file.with_suffix('.json')
                if json_file.exists():
                    json_file.unlink()
                    deleted_count += 1

                # Show progress every 1000 files
                if idx % 1000 == 0:
                    progress = (idx / total_to_delete) * 100
                    print(f"  Progress: {idx:,}/{total_to_delete:,} ({progress:.1f}%) - Deleted: {deleted_count:,}")

            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Show first 10 errors
                    print(f"  ❌ Error deleting {dup_file.name}: {e}")

        print()
        print("=" * 80)
        print("✅ DELETION COMPLETE")
        print("=" * 80)
        print()
        print(f"Successfully deleted:  {deleted_count:,} files")
        print(f"Errors:               {error_count:,}")
        print(f"Backup list:          {backup_file}")
        print()
        print("✅ Duplicate removal complete!")
        print()
        print("Run './check_status.sh' to see updated statistics.")
        print()

    print("=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
