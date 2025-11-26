#!/usr/bin/env python3
"""
Remove Images with Invalid or Missing JSON Files

This script removes image files that don't have valid corresponding JSON metadata files.
"""

import os
import json
from pathlib import Path

def remove_invalid_images(images_dir='./images', dry_run=True):
    """
    Remove images with invalid or missing JSON files

    Args:
        images_dir: Directory containing images
        dry_run: If True, only show what would be deleted without actually deleting
    """

    print("="*70)
    print("REMOVE IMAGES WITH INVALID/MISSING JSON FILES")
    print("="*70)
    print()

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(images_dir).rglob(ext))

    # Filter out macOS metadata files
    all_images = [img for img in all_images if not img.name.startswith('._')]

    print(f"Found {len(all_images)} image files in {images_dir}")
    print()

    # Find images with invalid or missing JSON
    invalid_images = []

    for img_path in all_images:
        json_path = img_path.with_suffix('.json')

        # Check if JSON file exists
        if not json_path.exists():
            invalid_images.append((img_path, "Missing JSON file"))
            continue

        # Check if JSON is valid
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

                # Check if JSON has required fields
                required_fields = ['filename', 'image_id', 'md5_hash']
                missing_fields = [field for field in required_fields if field not in data]

                if missing_fields:
                    invalid_images.append((img_path, f"Missing fields: {', '.join(missing_fields)}"))
        except json.JSONDecodeError as e:
            invalid_images.append((img_path, f"Invalid JSON: {e}"))
        except Exception as e:
            invalid_images.append((img_path, f"Error reading JSON: {e}"))

    print(f"Images with invalid/missing JSON: {len(invalid_images)}")
    print()

    if len(invalid_images) == 0:
        print("✅ All images have valid JSON metadata files!")
        return

    # Show what will be deleted
    print("The following images will be removed:")
    print()
    for idx, (img_path, reason) in enumerate(invalid_images[:10], 1):
        print(f"  {idx}. {img_path.name}")
        print(f"     Reason: {reason}")

    if len(invalid_images) > 10:
        print(f"  ... and {len(invalid_images) - 10} more")

    print()
    print("="*70)

    if dry_run:
        print("DRY RUN MODE - No files will be deleted")
        print()
        print("To actually delete these files, run:")
        print("  python3 remove_invalid_images.py --delete")
        return

    # Ask for confirmation
    print("⚠️  WARNING: This will permanently delete the image files!")
    print()
    response = input("Are you sure you want to delete these images? (type 'DELETE' to confirm): ").strip()

    if response != 'DELETE':
        print("Cancelled. No files were deleted.")
        return

    print()
    print("Deleting images...")
    print()

    deleted_count = 0
    error_count = 0

    for img_path, reason in invalid_images:
        try:
            # Delete the image file
            os.remove(img_path)

            # Also try to delete the JSON file if it exists
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                os.remove(json_path)

            print(f"✓ Deleted: {img_path.name}")
            deleted_count += 1

        except Exception as e:
            print(f"✗ Failed to delete {img_path.name}: {e}")
            error_count += 1

    print()
    print("="*70)
    print("DELETION COMPLETE")
    print("="*70)
    print(f"✓ Deleted: {deleted_count} images")
    print(f"✗ Errors: {error_count}")
    print()

    if deleted_count > 0:
        print("✅ Invalid images have been removed!")
        print()
        print("Remaining images should now process successfully.")

if __name__ == "__main__":
    import sys

    # Check if --delete flag is provided
    if '--delete' in sys.argv:
        print("⚠️  DELETE MODE ENABLED")
        print()
        remove_invalid_images(dry_run=False)
    else:
        print("Running in DRY RUN mode (no files will be deleted)")
        print()
        remove_invalid_images(dry_run=True)
