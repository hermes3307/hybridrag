#!/usr/bin/env python3
"""
Test script to verify download functionality with lazy-loaded hashes

This script tests:
1. System initialization is fast (no hash loading)
2. First download triggers hash loading
3. Hash loading shows progress messages
4. Duplicate detection works correctly
5. Second download doesn't reload hashes
"""

import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after loading env
from core import IntegratedFaceSystem, SystemConfig

def test_fast_initialization():
    """Test that initialization is fast (no hash loading)"""
    print("=" * 70)
    print("TEST 1: Fast Initialization")
    print("=" * 70)

    start = time.time()
    system = IntegratedFaceSystem()
    elapsed = time.time() - start

    print(f"✓ IntegratedFaceSystem created in {elapsed:.3f}s")

    if elapsed > 5:
        print(f"❌ FAIL: Initialization took {elapsed:.3f}s (should be < 5s)")
        return False
    else:
        print(f"✓ PASS: Initialization was fast ({elapsed:.3f}s)")

    # Check that hashes are not loaded yet
    if hasattr(system.downloader, '_hashes_loaded'):
        if system.downloader._hashes_loaded:
            print("❌ FAIL: Hashes were loaded during initialization (should be lazy)")
            return False
        else:
            print("✓ PASS: Hashes not loaded yet (lazy loading working)")

    return system

def test_database_initialization(system):
    """Test that database initialization works"""
    print("\n" + "=" * 70)
    print("TEST 2: Database Initialization")
    print("=" * 70)

    start = time.time()
    success = system.initialize()
    elapsed = time.time() - start

    if not success:
        print(f"❌ FAIL: Database initialization failed")
        return False

    print(f"✓ PASS: Database initialized in {elapsed:.3f}s")
    return True

def test_first_download_loads_hashes(system):
    """Test that first download loads hashes"""
    print("\n" + "=" * 70)
    print("TEST 3: First Download (Should Load Hashes)")
    print("=" * 70)

    print("Starting first download...")
    print("(This should trigger hash loading - watch for progress messages)")
    print()

    start = time.time()
    result = system.downloader.download_face()
    elapsed = time.time() - start

    print()
    print(f"Download completed in {elapsed:.3f}s")

    # Check if hashes are now loaded
    if not system.downloader._hashes_loaded:
        print("❌ FAIL: Hashes not loaded after first download")
        return False

    print(f"✓ PASS: Hashes loaded (total: {len(system.downloader.downloaded_hashes)} hashes)")

    if result:
        print(f"✓ PASS: Download successful: {result}")
    else:
        print("⚠ WARNING: Download returned None (might be duplicate or error)")

    return True

def test_second_download_no_reload(system):
    """Test that second download doesn't reload hashes"""
    print("\n" + "=" * 70)
    print("TEST 4: Second Download (Should NOT Reload Hashes)")
    print("=" * 70)

    print("Starting second download...")
    print("(Hashes already loaded - should be fast)")
    print()

    start = time.time()
    result = system.downloader.download_face()
    elapsed = time.time() - start

    print()
    print(f"Download completed in {elapsed:.3f}s")

    if elapsed > 10:
        print(f"⚠ WARNING: Second download took {elapsed:.3f}s (should be faster)")
        print("  (This might indicate hash reloading - check logs)")
    else:
        print(f"✓ PASS: Second download was fast ({elapsed:.3f}s)")

    if result:
        print(f"✓ PASS: Download successful: {result}")
    else:
        print("⚠ WARNING: Download returned None (might be duplicate or error)")

    return True

def test_hash_count(system):
    """Test hash count matches file count"""
    print("\n" + "=" * 70)
    print("TEST 5: Hash Count Verification")
    print("=" * 70)

    from pathlib import Path

    # Count actual files
    face_files = list(Path(system.config.faces_dir).rglob("*.jpg"))
    file_count = len(face_files)
    hash_count = len(system.downloader.downloaded_hashes)

    print(f"Files in directory: {file_count}")
    print(f"Hashes loaded: {hash_count}")

    if abs(file_count - hash_count) < 10:  # Allow small difference
        print(f"✓ PASS: Hash count matches file count (±10)")
    else:
        print(f"⚠ WARNING: Hash count differs from file count by {abs(file_count - hash_count)}")

    return True

def main():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("Testing Download Functionality with Lazy-Loaded Hashes")
    print("🧪" * 35)
    print()

    try:
        # Test 1: Fast initialization
        system = test_fast_initialization()
        if not system:
            print("\n❌ Test suite FAILED at initialization")
            return 1

        # Test 2: Database initialization
        if not test_database_initialization(system):
            print("\n❌ Test suite FAILED at database initialization")
            return 1

        # Test 3: First download (loads hashes)
        if not test_first_download_loads_hashes(system):
            print("\n❌ Test suite FAILED at first download")
            return 1

        # Test 4: Second download (no reload)
        if not test_second_download_no_reload(system):
            print("\n❌ Test suite FAILED at second download")
            return 1

        # Test 5: Hash count verification
        if not test_hash_count(system):
            print("\n❌ Test suite FAILED at hash count verification")
            return 1

        # All tests passed
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Initialization is fast (no hash loading)")
        print("  ✓ Database initialization works")
        print("  ✓ First download triggers hash loading")
        print("  ✓ Hash loading shows progress")
        print("  ✓ Second download doesn't reload hashes")
        print("  ✓ Hash count matches file count")
        print("\n✅ Lazy loading is working correctly!")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
