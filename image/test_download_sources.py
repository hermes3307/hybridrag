#!/usr/bin/env python3
"""
Test AI Image Download Sources

This script tests all configured AI image download sources to verify they're working
before running the main application.
"""

import requests
import time
from datetime import datetime

# Test configurations
TIMEOUT = 10  # seconds
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# All available download sources
DOWNLOAD_SOURCES = {
    'thispersondoesnotexist': {
        'name': 'ThisPersonDoesNotExist.com',
        'url': 'https://thispersondoesnotexist.com/',
        'description': 'High quality AI-generated faces (1024x1024)',
        'category': 'faces'
    },
    '100k-faces': {
        'name': '100K AI Faces',
        'url': 'https://100k-faces.vercel.app/api/random-image',
        'description': '100K AI-generated faces from generated.photos',
        'category': 'faces'
    },
    'thisartworkdoesnotexist': {
        'name': 'ThisArtworkDoesNotExist',
        'url': 'https://thisartworkdoesnotexist.com/',
        'description': 'AI-generated artwork and paintings',
        'category': 'artwork'
    },
    'thiscatdoesnotexist': {
        'name': 'ThisCatDoesNotExist',
        'url': 'https://thiscatdoesnotexist.com/',
        'description': 'GAN-generated cat images',
        'category': 'animals'
    },
    'thishorsedoesnotexist': {
        'name': 'ThisHorseDoesNotExist',
        'url': 'https://thishorsedoesnotexist.com/',
        'description': 'AI-generated horse images',
        'category': 'animals'
    },
    'unrealperson': {
        'name': 'UnrealPerson',
        'url': 'https://www.unrealperson.com/image',
        'description': 'Diverse AI-generated people and objects',
        'category': 'mixed'
    }
}

def test_source(source_key, source_info):
    """Test a single download source"""
    print(f"\n{'='*70}")
    print(f"Testing: {source_info['name']}")
    print(f"URL: {source_info['url']}")
    print(f"Category: {source_info['category']}")
    print(f"Description: {source_info['description']}")
    print(f"{'='*70}")

    try:
        # Make request
        start_time = time.time()
        response = requests.get(
            source_info['url'],
            headers={'User-Agent': USER_AGENT},
            timeout=TIMEOUT,
            allow_redirects=True
        )
        elapsed_time = time.time() - start_time

        # Check response
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '')

        # Check if it's an image
        is_image = any(img_type in content_type.lower() for img_type in ['image/', 'jpeg', 'jpg', 'png'])

        # Get file size
        file_size = len(response.content)
        file_size_kb = file_size / 1024
        file_size_mb = file_size_kb / 1024

        # Print results
        print(f"\n✅ SUCCESS!")
        print(f"   Status Code: {response.status_code}")
        print(f"   Content-Type: {content_type}")
        print(f"   Is Image: {'Yes' if is_image else 'No (WARNING!)'}")
        print(f"   File Size: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
        print(f"   Response Time: {elapsed_time:.2f}s")

        if is_image:
            print(f"   ✓ This source is working correctly!")
            return True, elapsed_time, file_size
        else:
            print(f"   ⚠️  WARNING: Response is not an image!")
            print(f"   Content preview: {response.content[:200]}")
            return False, elapsed_time, file_size

    except requests.exceptions.Timeout:
        print(f"\n❌ FAILED: Request timed out after {TIMEOUT}s")
        return False, TIMEOUT, 0

    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ FAILED: Connection error")
        print(f"   Error: {str(e)[:200]}")
        return False, 0, 0

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ FAILED: HTTP error")
        print(f"   Status Code: {response.status_code}")
        print(f"   Error: {str(e)[:200]}")
        return False, 0, 0

    except Exception as e:
        print(f"\n❌ FAILED: Unexpected error")
        print(f"   Error: {type(e).__name__}: {str(e)[:200]}")
        return False, 0, 0

def main():
    """Test all download sources"""
    print("="*70)
    print("AI IMAGE DOWNLOAD SOURCES - CONNECTION TEST")
    print("="*70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timeout: {TIMEOUT}s per source")
    print(f"Total sources to test: {len(DOWNLOAD_SOURCES)}")

    # Track results
    results = {}
    working_count = 0
    failed_count = 0
    total_time = 0
    total_size = 0

    # Test each source
    for source_key, source_info in DOWNLOAD_SOURCES.items():
        success, elapsed, size = test_source(source_key, source_info)
        results[source_key] = {
            'success': success,
            'elapsed': elapsed,
            'size': size,
            'name': source_info['name'],
            'category': source_info['category']
        }

        if success:
            working_count += 1
            total_time += elapsed
            total_size += size
        else:
            failed_count += 1

        # Small delay between tests
        time.sleep(0.5)

    # Print summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Sources Tested: {len(DOWNLOAD_SOURCES)}")
    print(f"✅ Working: {working_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"Success Rate: {(working_count/len(DOWNLOAD_SOURCES)*100):.1f}%")

    if working_count > 0:
        avg_time = total_time / working_count
        avg_size = total_size / working_count / 1024
        print(f"\nAverage Response Time: {avg_time:.2f}s")
        print(f"Average File Size: {avg_size:.2f} KB")

    # Print working sources by category
    print(f"\n{'='*70}")
    print("WORKING SOURCES BY CATEGORY")
    print(f"{'='*70}")

    categories = {}
    for source_key, result in results.items():
        if result['success']:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result['name'])

    for category, sources in sorted(categories.items()):
        print(f"\n{category.upper()}:")
        for source in sources:
            print(f"  ✓ {source}")

    # Print failed sources
    if failed_count > 0:
        print(f"\n{'='*70}")
        print("FAILED SOURCES")
        print(f"{'='*70}")
        for source_key, result in results.items():
            if not result['success']:
                print(f"  ✗ {result['name']} ({result['category']})")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    if working_count == 0:
        print("⚠️  WARNING: No sources are working!")
        print("   Check your internet connection and try again.")
    elif working_count < len(DOWNLOAD_SOURCES):
        print(f"✓ {working_count} source(s) working - you can proceed with these.")
        print("  Failed sources may be temporarily down or blocked.")
    else:
        print("✓ All sources are working! You're good to go!")

    print(f"\n{'='*70}")
    print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    return working_count > 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
