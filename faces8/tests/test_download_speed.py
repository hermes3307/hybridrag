#!/usr/bin/env python3
"""Test download speed from different face image sources"""

import requests
import time
from typing import Dict

def test_thispersondoesnotexist() -> Dict:
    """Test download speed from thispersondoesnotexist.com"""
    url = "https://thispersondoesnotexist.com/"
    headers = {'User-Agent': 'Mozilla/5.0'}

    times = []
    sizes = []

    print("\n[Testing thispersondoesnotexist.com]")
    for i in range(5):
        try:
            start = time.time()
            response = requests.get(url, headers=headers, timeout=30)
            elapsed = time.time() - start

            if response.status_code == 200:
                size_kb = len(response.content) / 1024
                times.append(elapsed)
                sizes.append(size_kb)
                speed_kbps = size_kb / elapsed
                print(f"  Test {i+1}: {elapsed:.2f}s, {size_kb:.1f} KB, {speed_kbps:.1f} KB/s")
            else:
                print(f"  Test {i+1}: Failed (HTTP {response.status_code})")
        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")

        time.sleep(0.5)

    if times:
        avg_time = sum(times) / len(times)
        avg_size = sum(sizes) / len(sizes)
        avg_speed = avg_size / avg_time
        return {
            'source': 'thispersondoesnotexist',
            'avg_time': avg_time,
            'avg_size_kb': avg_size,
            'avg_speed_kbps': avg_speed,
            'success_rate': len(times) / 5 * 100
        }
    return None

def test_100k_faces() -> Dict:
    """Test download speed from 100k-faces.vercel.app"""
    url = "https://100k-faces.vercel.app/api/random-image"
    headers = {'User-Agent': 'Mozilla/5.0'}

    times = []
    sizes = []

    print("\n[Testing 100k-faces.vercel.app]")
    for i in range(5):
        try:
            start = time.time()
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            elapsed = time.time() - start

            if response.status_code == 200:
                size_kb = len(response.content) / 1024
                times.append(elapsed)
                sizes.append(size_kb)
                speed_kbps = size_kb / elapsed
                print(f"  Test {i+1}: {elapsed:.2f}s, {size_kb:.1f} KB, {speed_kbps:.1f} KB/s")
            else:
                print(f"  Test {i+1}: Failed (HTTP {response.status_code})")
        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")

        time.sleep(0.5)

    if times:
        avg_time = sum(times) / len(times)
        avg_size = sum(sizes) / len(sizes)
        avg_speed = avg_size / avg_time
        return {
            'source': '100k-faces',
            'avg_time': avg_time,
            'avg_size_kb': avg_size,
            'avg_speed_kbps': avg_speed,
            'success_rate': len(times) / 5 * 100
        }
    return None

if __name__ == "__main__":
    print("=" * 60)
    print("FACE IMAGE DOWNLOAD SPEED TEST")
    print("=" * 60)

    result1 = test_thispersondoesnotexist()
    result2 = test_100k_faces()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if result1:
        print(f"\nthispersondoesnotexist.com:")
        print(f"  Average Time:  {result1['avg_time']:.2f}s")
        print(f"  Average Size:  {result1['avg_size_kb']:.1f} KB")
        print(f"  Average Speed: {result1['avg_speed_kbps']:.1f} KB/s")
        print(f"  Success Rate:  {result1['success_rate']:.0f}%")

    if result2:
        print(f"\n100k-faces.vercel.app:")
        print(f"  Average Time:  {result2['avg_time']:.2f}s")
        print(f"  Average Size:  {result2['avg_size_kb']:.1f} KB")
        print(f"  Average Speed: {result2['avg_speed_kbps']:.1f} KB/s")
        print(f"  Success Rate:  {result2['success_rate']:.0f}%")

    if result1 and result2:
        print("\n" + "-" * 60)
        if result1['avg_speed_kbps'] > result2['avg_speed_kbps']:
            print(f"⚡ WINNER: thispersondoesnotexist.com ({result1['avg_speed_kbps']:.1f} KB/s)")
        else:
            print(f"⚡ WINNER: 100k-faces.vercel.app ({result2['avg_speed_kbps']:.1f} KB/s)")

    print("=" * 60)
