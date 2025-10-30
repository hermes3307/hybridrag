#!/usr/bin/env python3
"""
Database Inspector Tool
Analyzes the ChromaDB database to show what metadata is actually saved
"""

import json
from core import IntegratedFaceSystem

def inspect_database():
    """Inspect the database and show sample metadata"""
    print("=" * 80)
    print("DATABASE INSPECTION TOOL")
    print("=" * 80)

    # Initialize system
    system = IntegratedFaceSystem()
    if not system.initialize():
        print("❌ Failed to initialize system")
        return

    # Get collection info
    db_info = system.db_manager.get_collection_info()
    print(f"\n📊 Collection: {db_info.get('name')}")
    print(f"📁 Path: {db_info.get('path')}")
    print(f"🔢 Total records: {db_info.get('count')}")

    if db_info.get('count', 0) == 0:
        print("\n⚠️  Database is empty. Download and process some faces first!")
        return

    # Get sample records
    print("\n" + "=" * 80)
    print("SAMPLE RECORDS (showing first 5)")
    print("=" * 80)

    try:
        # Get first 5 records
        results = system.db_manager.collection.get(limit=5)

        if not results or not results.get('metadatas'):
            print("No metadata found")
            return

        # Collect all unique metadata keys
        all_keys = set()
        for metadata in results['metadatas']:
            all_keys.update(metadata.keys())

        print(f"\n🔑 AVAILABLE METADATA FIELDS ({len(all_keys)} total):")
        print("-" * 80)

        # Categorize fields
        demographics = []
        image_props = []
        system_fields = []
        other = []

        for key in sorted(all_keys):
            if key in ['estimated_sex', 'sex', 'age_group', 'estimated_age', 'skin_tone',
                       'skin_color', 'hair_color']:
                demographics.append(key)
            elif key in ['width', 'height', 'brightness', 'contrast', 'saturation_mean',
                        'format', 'mode', 'size_bytes', 'faces_detected', 'brightness_level',
                        'image_quality', 'has_face']:
                image_props.append(key)
            elif key in ['face_id', 'file_path', 'timestamp', 'image_hash', 'embedding_model']:
                system_fields.append(key)
            else:
                other.append(key)

        if demographics:
            print("\n👤 DEMOGRAPHIC FIELDS (searchable for finding specific people):")
            for key in demographics:
                values = set(str(m.get(key, 'N/A')) for m in results['metadatas'])
                print(f"   • {key:20s} → {', '.join(sorted(values))}")

        if image_props:
            print("\n🖼️  IMAGE PROPERTY FIELDS (searchable for image quality/characteristics):")
            for key in image_props:
                sample_val = results['metadatas'][0].get(key, 'N/A')
                print(f"   • {key:20s} → {sample_val}")

        if system_fields:
            print("\n⚙️  SYSTEM FIELDS (for tracking and identification):")
            for key in system_fields:
                sample_val = str(results['metadatas'][0].get(key, 'N/A'))
                if len(sample_val) > 50:
                    sample_val = sample_val[:50] + "..."
                print(f"   • {key:20s} → {sample_val}")

        if other:
            print("\n📋 OTHER FIELDS:")
            for key in other:
                print(f"   • {key}")

        # Show sample full record
        print("\n" + "=" * 80)
        print("📄 FULL SAMPLE RECORD (record #1):")
        print("=" * 80)
        print(json.dumps(results['metadatas'][0], indent=2, default=str))

        # Analyze demographic distribution
        print("\n" + "=" * 80)
        print("📈 DEMOGRAPHIC DISTRIBUTION (all records)")
        print("=" * 80)

        all_records = system.db_manager.collection.get(limit=10000)

        if all_records and all_records.get('metadatas'):
            # Count demographics
            sex_counts = {}
            age_counts = {}
            skin_tone_counts = {}
            hair_color_counts = {}

            for metadata in all_records['metadatas']:
                # Sex
                sex = metadata.get('estimated_sex') or metadata.get('sex', 'unknown')
                sex_counts[sex] = sex_counts.get(sex, 0) + 1

                # Age
                age = metadata.get('age_group', 'unknown')
                age_counts[age] = age_counts.get(age, 0) + 1

                # Skin tone
                skin = metadata.get('skin_tone', 'unknown')
                skin_tone_counts[skin] = skin_tone_counts.get(skin, 0) + 1

                # Hair color
                hair = metadata.get('hair_color', 'unknown')
                hair_color_counts[hair] = hair_color_counts.get(hair, 0) + 1

            print(f"\n👤 Sex Distribution:")
            for sex, count in sorted(sex_counts.items(), key=lambda x: -x[1]):
                pct = (count / len(all_records['metadatas']) * 100)
                print(f"   {sex:15s}: {count:4d} ({pct:5.1f}%)")

            print(f"\n🎂 Age Group Distribution:")
            for age, count in sorted(age_counts.items(), key=lambda x: -x[1]):
                pct = (count / len(all_records['metadatas']) * 100)
                print(f"   {age:15s}: {count:4d} ({pct:5.1f}%)")

            print(f"\n🎨 Skin Tone Distribution:")
            for skin, count in sorted(skin_tone_counts.items(), key=lambda x: -x[1]):
                pct = (count / len(all_records['metadatas']) * 100)
                print(f"   {skin:15s}: {count:4d} ({pct:5.1f}%)")

            print(f"\n💇 Hair Color Distribution:")
            for hair, count in sorted(hair_color_counts.items(), key=lambda x: -x[1]):
                pct = (count / len(all_records['metadatas']) * 100)
                print(f"   {hair:15s}: {count:4d} ({pct:5.1f}%)")

        # Show example queries
        print("\n" + "=" * 80)
        print("💡 EXAMPLE SEARCH QUERIES YOU CAN RUN:")
        print("=" * 80)
        print("""
1. Find all females with blonde hair:
   {'estimated_sex': 'female', 'hair_color': 'blonde'}

2. Find all adults with dark skin:
   {'age_group': 'adult', 'skin_color': 'dark'}

3. Find high quality bright images:
   {'image_quality': 'high', 'brightness_level': 'bright'}

4. Find young adults:
   {'age_group': 'young_adult'}

5. Find people with red hair:
   {'hair_color': 'red'}

6. Find images with detected faces:
   {'has_face': True}
        """)

    except Exception as e:
        print(f"\n❌ Error inspecting database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_database()
