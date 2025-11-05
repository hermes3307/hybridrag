#!/usr/bin/env python3
"""
Test the updated downloader with new Picsum sources
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core import ImageDownloader, SystemConfig, SystemStats

def test_downloader():
    """Test downloading from new sources"""
    print("="*60)
    print("TESTING UPDATED IMAGE DOWNLOADER")
    print("="*60)

    # Create test directory
    test_dir = "./test_downloads"
    os.makedirs(test_dir, exist_ok=True)

    # Test each source
    for source_key, source_info in ImageDownloader.DOWNLOAD_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"Testing: {source_info['name']}")
        print(f"Category: {source_info['category']}")
        print(f"URL: {source_info['url']}")
        print(f"{'='*60}")

        # Configure for this source
        config = SystemConfig()
        config.images_dir = test_dir
        config.download_source = source_key
        config.download_delay = 0.5

        stats = SystemStats()
        downloader = ImageDownloader(config, stats)

        # Try downloading
        try:
            file_path = downloader.download_image()

            if file_path:
                print(f"✅ SUCCESS!")
                print(f"   Downloaded to: {file_path}")

                # Check file size
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   File size: {size/1024:.1f} KB")

                    # Check if JSON metadata was created
                    json_path = file_path.replace('.jpg', '.json')
                    if os.path.exists(json_path):
                        print(f"   ✓ Metadata file created")
                else:
                    print(f"   ⚠️  File not found on disk!")
            else:
                print(f"❌ FAILED: Download returned None")

        except Exception as e:
            print(f"❌ ERROR: {str(e)[:200]}")

    # Summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    stats_info = stats.get_stats()
    print(f"Download attempts: {stats_info['download_attempts']}")
    print(f"Successful: {stats_info['download_success']}")
    print(f"Duplicates: {stats_info['download_duplicates']}")
    print(f"Errors: {stats_info['download_errors']}")

    if stats_info['download_success'] > 0:
        print(f"\n✅ Downloader is working!")
        print(f"Test images saved in: {test_dir}/")
    else:
        print(f"\n❌ No successful downloads")

    return stats_info['download_success'] > 0

if __name__ == '__main__':
    success = test_downloader()
    exit(0 if success else 1)
