#!/usr/bin/env python3
"""
Search Examples - Interactive Demo

This script provides interactive examples of the advanced search functionality.
Run this after you have downloaded and processed some faces.
"""

from core import IntegratedFaceSystem
from advanced_search import AdvancedSearchEngine, SearchQuery
import json
from pathlib import Path


def example_1_basic_metadata_search(search_engine):
    """Example 1: Basic metadata search - Find blonde females"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Metadata Search")
    print("=" * 80)
    print("Goal: Find all blonde females in the database\n")

    query = SearchQuery(
        sex=['female'],
        hair_colors=['blonde'],
        n_results=5,
        search_mode='metadata'
    )

    print(f"Query: sex=female, hair_color=blonde")
    results = search_engine.search(query)

    print(f"\n✅ Found {len(results)} results:")
    for i, result in enumerate(results[:5], 1):
        metadata = result['metadata']
        print(f"  {i}. {Path(metadata['file_path']).name} - "
              f"{metadata.get('estimated_sex')} - "
              f"{metadata.get('age_group')} - "
              f"{metadata.get('hair_color')} hair")


def example_2_multiple_filters_or_logic(search_engine):
    """Example 2: Multiple values with OR logic"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Values with OR Logic")
    print("=" * 80)
    print("Goal: Find people who are young_adult OR adult with brown OR black hair\n")

    query = SearchQuery(
        age_groups=['young_adult', 'adult'],  # OR logic
        hair_colors=['brown', 'black'],       # OR logic
        n_results=5,
        search_mode='metadata'
    )

    print("Query: age=(young_adult OR adult) AND hair=(brown OR black)")
    results = search_engine.search(query)

    print(f"\n✅ Found {len(results)} results:")
    for i, result in enumerate(results[:5], 1):
        metadata = result['metadata']
        print(f"  {i}. {metadata.get('age_group'):15s} - "
              f"{metadata.get('hair_color'):15s} - "
              f"{metadata.get('estimated_sex')}")


def example_3_text_based_search(search_engine):
    """Example 3: Natural language text search"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Text-Based Natural Language Search")
    print("=" * 80)
    print("Goal: Use natural language to search\n")

    queries = [
        "blonde female",
        "young adult male with dark hair",
        "seniors with gray hair"
    ]

    for query_text in queries:
        print(f"\n🔍 Query: '{query_text}'")
        results = search_engine.text_search(query_text, n_results=3)

        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                metadata = result['metadata']
                print(f"      {i}. {metadata.get('estimated_sex')} - "
                      f"{metadata.get('age_group')} - "
                      f"{metadata.get('hair_color')} hair")
        else:
            print("   No results found")


def example_4_range_queries(search_engine):
    """Example 4: Range queries for brightness"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Range Queries")
    print("=" * 80)
    print("Goal: Find images with specific brightness range\n")

    # Bright images
    query = SearchQuery(
        brightness_range=(180, 255),  # Very bright
        n_results=3,
        search_mode='metadata'
    )

    print("Query: brightness 180-255 (bright images)")
    results = search_engine.search(query)

    print(f"\n✅ Found {len(results)} bright images:")
    for i, result in enumerate(results[:3], 1):
        metadata = result['metadata']
        brightness = metadata.get('brightness', 0)
        print(f"  {i}. Brightness: {brightness:.1f} - {Path(metadata['file_path']).name}")


def example_5_complex_query(search_engine):
    """Example 5: Complex multi-filter query"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complex Multi-Filter Query")
    print("=" * 80)
    print("Goal: Combine multiple filters for precise results\n")

    query = SearchQuery(
        sex=['female'],
        age_groups=['young_adult', 'adult'],
        skin_colors=['light', 'medium'],
        hair_colors=['blonde', 'brown'],
        has_face=True,
        brightness_range=(100, 200),
        n_results=5,
        search_mode='metadata'
    )

    print("Query:")
    print("  - Sex: female")
    print("  - Age: young_adult OR adult")
    print("  - Skin: light OR medium")
    print("  - Hair: blonde OR brown")
    print("  - Has Face: yes")
    print("  - Brightness: 100-200")

    results = search_engine.search(query)

    print(f"\n✅ Found {len(results)} results matching ALL criteria:")
    for i, result in enumerate(results[:5], 1):
        metadata = result['metadata']
        print(f"  {i}. {metadata.get('estimated_sex'):8s} - "
              f"{metadata.get('age_group'):15s} - "
              f"{metadata.get('skin_color'):8s} - "
              f"{metadata.get('hair_color'):12s} - "
              f"brightness: {metadata.get('brightness', 0):.0f}")


def example_6_demographic_distribution(search_engine):
    """Example 6: Analyze demographic distribution"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Database Statistics & Demographics")
    print("=" * 80)
    print("Goal: Understand the composition of your face database\n")

    stats = search_engine.get_statistics()

    if not stats or stats.get('total_count', 0) == 0:
        print("❌ No data in database yet. Download and process some faces first!")
        return

    print(f"📊 Total Faces: {stats['total_count']}\n")

    print("👤 Sex Distribution:")
    for sex, count in sorted(stats['sex_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {sex:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    print("\n🎂 Age Distribution:")
    for age, count in sorted(stats['age_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {age:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    print("\n💇 Hair Color Distribution:")
    for hair, count in sorted(stats['hair_color_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {hair:15s}: {count:4d} ({pct:5.1f}%) {bar}")


def example_7_save_and_export(search_engine):
    """Example 7: Save queries and export results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Save Queries & Export Results")
    print("=" * 80)
    print("Goal: Save frequently-used queries and export results\n")

    # Create a useful query
    query = SearchQuery(
        sex=['female'],
        age_groups=['young_adult'],
        n_results=10,
        search_mode='metadata'
    )

    # Save it
    search_engine.save_query("young_females", query)
    print("✅ Saved query 'young_females'")

    # Load and use it
    loaded_query = search_engine.load_query("young_females")
    results = search_engine.search(loaded_query)

    print(f"✅ Loaded and executed saved query: found {len(results)} results")

    # Export results to different formats
    if results:
        search_engine.export_results(results, "example_output.json", format='json')
        search_engine.export_results(results, "example_output.csv", format='csv')
        search_engine.export_results(results, "example_output.txt", format='txt')

        print("\n💾 Exported results to:")
        print("   - example_output.json")
        print("   - example_output.csv")
        print("   - example_output.txt")

    # List all saved queries
    saved = search_engine.list_saved_queries()
    print(f"\n📋 Saved queries: {', '.join(saved)}")


def example_8_demographic_dataset_builder(search_engine):
    """Example 8: Build demographic-specific datasets"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Build Demographic-Specific Datasets")
    print("=" * 80)
    print("Goal: Create separate datasets for different demographics\n")

    demographics = [
        ('males', SearchQuery(sex=['male'], n_results=100)),
        ('females', SearchQuery(sex=['female'], n_results=100)),
        ('young_adults', SearchQuery(age_groups=['young_adult'], n_results=100)),
        ('seniors', SearchQuery(age_groups=['senior'], n_results=100)),
    ]

    print("Building demographic datasets...")
    for name, query in demographics:
        query.search_mode = 'metadata'
        results = search_engine.search(query)

        if results:
            filename = f"dataset_{name}.json"
            search_engine.export_results(results, filename, format='json')
            print(f"  ✅ {name:20s}: {len(results):3d} faces → {filename}")
        else:
            print(f"  ⚠️  {name:20s}: No results found")


def main():
    """Run all examples"""
    print("\n")
    print("=" * 80)
    print("ADVANCED SEARCH EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates the advanced search capabilities.")
    print("Make sure you have downloaded and processed faces first!")
    print("\nStarting system initialization...")

    # Initialize system
    system = IntegratedFaceSystem()
    if not system.initialize():
        print("\n❌ Failed to initialize system")
        print("Make sure ChromaDB is installed: pip install chromadb")
        return

    search_engine = AdvancedSearchEngine(system)

    # Check if database has data
    db_info = system.db_manager.get_collection_info()
    count = db_info.get('count', 0)

    if count == 0:
        print("\n" + "=" * 80)
        print("⚠️  DATABASE IS EMPTY!")
        print("=" * 80)
        print("\nTo use these examples, you need to:")
        print("1. Start the GUI: python faces.py")
        print("2. Download faces: Go to 'Download Faces' tab → Click 'Start Download'")
        print("3. Process faces: Go to 'Process & Embed' tab → Click 'Process All Faces'")
        print("\nOr run the quickstart:")
        print("   python quickstart.py")
        print("\n" + "=" * 80)
        return

    print(f"\n✅ System initialized successfully")
    print(f"📊 Database contains {count} faces\n")

    # Run examples
    try:
        example_1_basic_metadata_search(search_engine)
        input("\nPress Enter to continue to Example 2...")

        example_2_multiple_filters_or_logic(search_engine)
        input("\nPress Enter to continue to Example 3...")

        example_3_text_based_search(search_engine)
        input("\nPress Enter to continue to Example 4...")

        example_4_range_queries(search_engine)
        input("\nPress Enter to continue to Example 5...")

        example_5_complex_query(search_engine)
        input("\nPress Enter to continue to Example 6...")

        example_6_demographic_distribution(search_engine)
        input("\nPress Enter to continue to Example 7...")

        example_7_save_and_export(search_engine)
        input("\nPress Enter to continue to Example 8...")

        example_8_demographic_dataset_builder(search_engine)

        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Try the CLI: python search_cli.py --help")
        print("  2. Read the guide: cat SEARCH_GUIDE.md")
        print("  3. Build your own searches using the examples above")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
