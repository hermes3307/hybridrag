#!/usr/bin/env python3
"""
Command-Line Interface for Advanced Search

Usage:
    python search_cli.py --text "blonde female"
    python search_cli.py --sex female --hair blonde --age young_adult
    python search_cli.py --stats
    python search_cli.py --list-values
"""

import argparse
from core import IntegratedFaceSystem
from advanced_search import AdvancedSearchEngine, SearchQuery
import json
from pathlib import Path


def print_results(results):
    """Pretty print search results"""
    if not results:
        print("❌ No results found")
        return

    print(f"\n✅ Found {len(results)} results:\n")
    print("=" * 100)

    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        distance = result.get('distance', 0)

        print(f"\n#{i} - Distance: {distance:.4f}")
        print(f"   📁 File: {Path(metadata.get('file_path', 'N/A')).name}")
        print(f"   🆔 ID: {result.get('id', 'N/A')}")

        # Demographics
        sex = metadata.get('estimated_sex', 'unknown')
        age_group = metadata.get('age_group', 'unknown')
        age_range = metadata.get('estimated_age', 'unknown')
        skin_tone = metadata.get('skin_tone', 'unknown')
        hair_color = metadata.get('hair_color', 'unknown')

        print(f"   👤 Demographics:")
        print(f"      Sex: {sex}")
        print(f"      Age: {age_group} ({age_range})")
        print(f"      Skin: {skin_tone}")
        print(f"      Hair: {hair_color}")

        # Image properties
        width = metadata.get('width', 'N/A')
        height = metadata.get('height', 'N/A')
        brightness = metadata.get('brightness', 'N/A')
        quality = metadata.get('image_quality', 'N/A')

        print(f"   🖼️  Image:")
        print(f"      Size: {width}x{height}")
        print(f"      Brightness: {brightness}")
        print(f"      Quality: {quality}")

        print("-" * 100)


def show_statistics(search_engine):
    """Show database statistics"""
    print("\n" + "=" * 80)
    print("📊 DATABASE STATISTICS")
    print("=" * 80)

    stats = search_engine.get_statistics()

    if not stats:
        print("❌ No data in database")
        return

    print(f"\n📈 Total Records: {stats['total_count']}")

    # Sex distribution
    print(f"\n👤 Sex Distribution:")
    for sex, count in sorted(stats['sex_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {sex:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Age distribution
    print(f"\n🎂 Age Distribution:")
    for age, count in sorted(stats['age_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {age:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Skin tone distribution
    print(f"\n🎨 Skin Tone Distribution:")
    for skin, count in sorted(stats['skin_tone_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {skin:15s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Hair color distribution
    print(f"\n💇 Hair Color Distribution:")
    for hair, count in sorted(stats['hair_color_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_count'] * 100)
        bar = '█' * int(pct / 2)
        print(f"   {hair:15s}: {count:4d} ({pct:5.1f}%) {bar}")


def list_possible_values():
    """List all possible search values"""
    print("\n" + "=" * 80)
    print("🔍 SEARCHABLE VALUES")
    print("=" * 80)

    print("\n👤 Sex:")
    print("   • male, female, unknown")

    print("\n🎂 Age Groups:")
    print("   • child (0-12)")
    print("   • young_adult (18-25)")
    print("   • adult (25-40)")
    print("   • middle_aged (40-60)")
    print("   • senior (60+)")

    print("\n🎨 Skin Tones:")
    print("   • very_light, light, medium, tan, brown, dark")

    print("\n🎨 Skin Colors (broad):")
    print("   • light, medium, dark")

    print("\n💇 Hair Colors:")
    print("   • black, dark_brown, brown, blonde, red, gray, light_gray, other")

    print("\n🖼️ Image Quality:")
    print("   • high, medium")

    print("\n💡 Brightness Level:")
    print("   • bright, dark")

    print("\n" + "=" * 80)
    print("💡 EXAMPLE QUERIES:")
    print("=" * 80)
    print("""
1. Text search:
   python search_cli.py --text "blonde female"
   python search_cli.py --text "young adult males with dark hair"

2. Multiple filters (OR logic):
   python search_cli.py --sex male female --hair brown black
   → Finds: (male OR female) AND (brown hair OR black hair)

3. Complex filters:
   python search_cli.py --sex female --age young_adult adult --hair blonde
   → Finds: females who are (young_adult OR adult) with blonde hair

4. Range filters:
   python search_cli.py --brightness-min 150 --brightness-max 255
   → Finds: bright images (brightness 150-255)

5. Image-based search with filters:
   python search_cli.py --image path/to/face.jpg --sex female
   → Finds: similar to image BUT only females
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Face Search CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Text search
    parser.add_argument('--text', type=str, help='Natural language text search')

    # Demographic filters (support multiple values for OR logic)
    parser.add_argument('--sex', nargs='+', choices=['male', 'female', 'unknown'],
                       help='Sex filter (can specify multiple for OR logic)')
    parser.add_argument('--age', nargs='+',
                       choices=['child', 'young_adult', 'adult', 'middle_aged', 'senior'],
                       help='Age group filter (can specify multiple for OR logic)')
    parser.add_argument('--skin-tone', nargs='+',
                       choices=['very_light', 'light', 'medium', 'tan', 'brown', 'dark'],
                       help='Skin tone filter (can specify multiple for OR logic)')
    parser.add_argument('--skin-color', nargs='+', choices=['light', 'medium', 'dark'],
                       help='Skin color filter (can specify multiple for OR logic)')
    parser.add_argument('--hair', nargs='+',
                       choices=['black', 'dark_brown', 'brown', 'blonde', 'red', 'gray', 'light_gray', 'other'],
                       help='Hair color filter (can specify multiple for OR logic)')

    # Image property filters
    parser.add_argument('--has-face', action='store_true', help='Only images with detected faces')
    parser.add_argument('--quality', choices=['high', 'medium'], help='Minimum image quality')
    parser.add_argument('--brightness-min', type=float, help='Minimum brightness (0-255)')
    parser.add_argument('--brightness-max', type=float, help='Maximum brightness (0-255)')

    # Image-based search
    parser.add_argument('--image', type=str, help='Query image path for similarity search')

    # Query options
    parser.add_argument('--limit', type=int, default=10, help='Number of results (default: 10)')
    parser.add_argument('--mode', choices=['vector', 'metadata', 'hybrid'], default='metadata',
                       help='Search mode (default: metadata)')
    parser.add_argument('--sort', choices=['distance', 'timestamp', 'brightness'],
                       help='Sort results by field')

    # Export
    parser.add_argument('--export', type=str, help='Export results to file (json, csv, or txt)')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json',
                       help='Export format (default: json)')

    # Utility commands
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--list-values', action='store_true', help='List all possible search values')

    args = parser.parse_args()

    # Show help commands
    if args.stats or args.list_values:
        system = IntegratedFaceSystem()
        if not system.initialize():
            print("❌ Failed to initialize system")
            return

        search_engine = AdvancedSearchEngine(system)

        if args.stats:
            show_statistics(search_engine)
        if args.list_values:
            list_possible_values()
        return

    # Build search query
    if args.text:
        # Text-based search
        print(f"🔍 Text Search: '{args.text}'")
        system = IntegratedFaceSystem()
        if not system.initialize():
            print("❌ Failed to initialize system")
            return

        search_engine = AdvancedSearchEngine(system)
        results = search_engine.text_search(args.text, n_results=args.limit)
        print_results(results)

        if args.export:
            search_engine.export_results(results, args.export, format=args.format)
            print(f"\n💾 Results exported to: {args.export}")

    else:
        # Build structured query
        query = SearchQuery()
        query.n_results = args.limit
        query.search_mode = args.mode

        if args.image:
            query.query_image = args.image
            if args.mode == 'metadata':
                query.search_mode = 'hybrid'  # Auto-switch to hybrid if image provided

        if args.sex:
            query.sex = args.sex
        if args.age:
            query.age_groups = args.age
        if args.skin_tone:
            query.skin_tones = args.skin_tone
        if args.skin_color:
            query.skin_colors = args.skin_color
        if args.hair:
            query.hair_colors = args.hair
        if args.has_face:
            query.has_face = True
        if args.quality:
            query.min_image_quality = args.quality
        if args.brightness_min is not None and args.brightness_max is not None:
            query.brightness_range = (args.brightness_min, args.brightness_max)
        if args.sort:
            query.sort_by = args.sort

        # Execute search
        print(f"🔍 Searching with filters:")
        if query.sex:
            print(f"   Sex: {', '.join(query.sex)}")
        if query.age_groups:
            print(f"   Age: {', '.join(query.age_groups)}")
        if query.hair_colors:
            print(f"   Hair: {', '.join(query.hair_colors)}")
        if query.skin_tones:
            print(f"   Skin Tone: {', '.join(query.skin_tones)}")
        if query.query_image:
            print(f"   Query Image: {query.query_image}")

        system = IntegratedFaceSystem()
        if not system.initialize():
            print("❌ Failed to initialize system")
            return

        search_engine = AdvancedSearchEngine(system)
        results = search_engine.search(query)
        print_results(results)

        if args.export:
            search_engine.export_results(results, args.export, format=args.format)
            print(f"\n💾 Results exported to: {args.export}")


if __name__ == "__main__":
    main()
