#!/usr/bin/env python3
"""
Download Sample Group Photos for Multi-Face Testing

Downloads free stock photos with multiple people from Unsplash
Requires: pip install requests
"""

import requests
import os
from pathlib import Path

# Sample Unsplash photo IDs with multiple people
# These are free to use and contain multiple faces
SAMPLE_PHOTOS = [
    # Groups of friends/people
    {
        'url': 'https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800',
        'name': 'group_friends_1.jpg',
        'description': '4-5 people smiling together'
    },
    {
        'url': 'https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800',
        'name': 'group_friends_2.jpg',
        'description': 'Group of friends casual photo'
    },
    {
        'url': 'https://images.unsplash.com/photo-1543269865-cbf427effbad?w=800',
        'name': 'team_photo.jpg',
        'description': 'Business team photo'
    },
    {
        'url': 'https://images.unsplash.com/photo-1521737711867-e3b97375f902?w=800',
        'name': 'family_group.jpg',
        'description': 'Family group photo'
    },
    {
        'url': 'https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=800',
        'name': 'conference_people.jpg',
        'description': 'People at conference'
    },
    {
        'url': 'https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=800',
        'name': 'team_meeting.jpg',
        'description': 'Team in meeting'
    },
    {
        'url': 'https://images.unsplash.com/photo-1543269664-56d93c1b41a6?w=800',
        'name': 'diverse_group.jpg',
        'description': 'Diverse group of people'
    },
    {
        'url': 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=800',
        'name': 'startup_team.jpg',
        'description': 'Startup team photo'
    }
]

def download_photo(url: str, save_path: str, description: str) -> bool:
    """Download a photo from URL"""
    try:
        print(f"Downloading: {description}")
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"  ✅ Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    # Create directory for test photos
    test_dir = Path("test_group_photos")
    test_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("DOWNLOADING SAMPLE GROUP PHOTOS")
    print("=" * 80)
    print(f"Target directory: {test_dir.absolute()}")
    print(f"Photos to download: {len(SAMPLE_PHOTOS)}")
    print()

    success_count = 0

    for photo in SAMPLE_PHOTOS:
        save_path = test_dir / photo['name']

        # Skip if already exists
        if save_path.exists():
            print(f"⏭️  Skipping: {photo['name']} (already exists)")
            success_count += 1
            continue

        if download_photo(photo['url'], str(save_path), photo['description']):
            success_count += 1

    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Successfully downloaded: {success_count}/{len(SAMPLE_PHOTOS)}")
    print(f"Location: {test_dir.absolute()}")
    print()
    print("Next steps:")
    print("1. Go to the 🎯 Advanced Search tab in the web interface")
    print("2. Upload one of these photos")
    print("3. Click 'Detect & Search All Faces'")
    print()

if __name__ == '__main__':
    main()
