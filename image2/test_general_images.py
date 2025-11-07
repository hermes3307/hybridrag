#!/usr/bin/env python3
"""
Test General/Landscape Image Sources (Not Faces)
"""

import requests
import time

TIMEOUT = 10
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Test sources for general/landscape images
TEST_SOURCES = {
    'picsum': {
        'name': 'Lorem Picsum (Unsplash)',
        'url': 'https://picsum.photos/1024/768',
        'description': 'Random high-quality photos from Unsplash',
        'category': 'general'
    },
    'picsum_landscape': {
        'name': 'Picsum Landscape',
        'url': 'https://picsum.photos/1920/1080',
        'description': 'Landscape format random photos',
        'category': 'landscape'
    },
    'unsplash_random': {
        'name': 'Unsplash Random',
        'url': 'https://source.unsplash.com/random/1600x900',
        'description': 'Random Unsplash photo',
        'category': 'general'
    },
    'unsplash_nature': {
        'name': 'Unsplash Nature',
        'url': 'https://source.unsplash.com/random/1600x900/?nature',
        'description': 'Random nature photos',
        'category': 'nature'
    },
    'unsplash_landscape': {
        'name': 'Unsplash Landscape',
        'url': 'https://source.unsplash.com/random/1920x1080/?landscape',
        'description': 'Random landscape photos',
        'category': 'landscape'
    },
    'unsplash_city': {
        'name': 'Unsplash City',
        'url': 'https://source.unsplash.com/random/1600x900/?city',
        'description': 'Random city/urban photos',
        'category': 'urban'
    },
    'unsplash_architecture': {
        'name': 'Unsplash Architecture',
        'url': 'https://source.unsplash.com/random/1600x900/?architecture',
        'description': 'Random architecture photos',
        'category': 'architecture'
    },
    'unsplash_animals': {
        'name': 'Unsplash Animals',
        'url': 'https://source.unsplash.com/random/1600x900/?animals',
        'description': 'Random animal photos',
        'category': 'animals'
    }
}

def test_source(source_key, source_info):
    """Test a single source"""
    print(f"\n{'='*60}")
    print(f"Testing: {source_info['name']}")
    print(f"URL: {source_info['url']}")
    print(f"{'='*60}")

    try:
        start = time.time()
        response = requests.get(
            source_info['url'],
            headers={'User-Agent': USER_AGENT},
            timeout=TIMEOUT,
            allow_redirects=True
        )
        elapsed = time.time() - start

        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        is_image = any(t in content_type.lower() for t in ['image/', 'jpeg', 'jpg', 'png'])
        size_kb = len(response.content) / 1024

        print(f"✅ SUCCESS!")
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {content_type}")
        print(f"   Is Image: {'Yes' if is_image else 'No'}")
        print(f"   Size: {size_kb:.1f} KB")
        print(f"   Time: {elapsed:.2f}s")

        return is_image, elapsed, size_kb

    except Exception as e:
        print(f"❌ FAILED: {str(e)[:100]}")
        return False, 0, 0

def main():
    print("="*60)
    print("TESTING GENERAL/LANDSCAPE IMAGE SOURCES")
    print("="*60)

    working = []
    failed = []

    for key, info in TEST_SOURCES.items():
        success, elapsed, size = test_source(key, info)
        if success:
            working.append((key, info, elapsed, size))
        else:
            failed.append((key, info))
        time.sleep(0.5)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Working: {len(working)}/{len(TEST_SOURCES)}")
    print(f"Failed: {len(failed)}/{len(TEST_SOURCES)}")

    if working:
        print(f"\n✅ WORKING SOURCES:")
        for key, info, elapsed, size in working:
            print(f"   {info['name']}")
            print(f"      Category: {info['category']}")
            print(f"      Speed: {elapsed:.2f}s, Size: {size:.1f} KB")

    if failed:
        print(f"\n❌ FAILED SOURCES:")
        for key, info in failed:
            print(f"   {info['name']}")

    return len(working) > 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
