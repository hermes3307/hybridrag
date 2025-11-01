#!/usr/bin/env python3
"""
Generate Missing Metadata for Face Images
Scans directory for JPG files without JSON metadata and generates them
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from PIL import Image

# Try to import face analysis
try:
    from core import FaceAnalyzer
    FACE_ANALYSIS_AVAILABLE = True
except ImportError:
    FACE_ANALYSIS_AVAILABLE = False
    print("Warning: FaceAnalyzer not available. Metadata will be basic.")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


class MetadataGenerator:
    """Generate metadata for existing face images"""

    def __init__(self, directory: str = "./faces", force: bool = False, auto_confirm: bool = False):
        self.directory = Path(directory)
        self.force = force  # Regenerate even if JSON exists
        self.auto_confirm = auto_confirm  # Skip confirmation prompts
        self.console = Console()

        # Initialize face analyzer if available
        self.face_analyzer = None
        if FACE_ANALYSIS_AVAILABLE:
            try:
                self.face_analyzer = FaceAnalyzer()
                self.console.print("[green]✓[/green] Face analyzer initialized")
            except Exception as e:
                self.console.print(f"[yellow]Warning:[/yellow] Could not initialize face analyzer: {e}")
        else:
            self.console.print("[yellow]![/yellow] Face analysis not available - generating basic metadata only")

        # Statistics
        self.stats = {
            'total_images': 0,
            'already_have_json': 0,
            'generated': 0,
            'errors': 0
        }

    def find_images_without_json(self) -> List[Path]:
        """Find all JPG images without corresponding JSON files"""

        if not self.directory.exists():
            self.console.print(f"[red]Error:[/red] Directory not found: {self.directory}")
            return []

        all_images = list(self.directory.glob("*.jpg")) + list(self.directory.glob("*.jpeg"))
        self.stats['total_images'] = len(all_images)

        images_without_json = []

        for img_path in all_images:
            # Determine expected JSON filename
            json_path = img_path.with_suffix('.json')

            if json_path.exists() and not self.force:
                self.stats['already_have_json'] += 1
            else:
                images_without_json.append(img_path)

        return images_without_json

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def generate_metadata_for_image(self, image_path: Path) -> bool:
        """Generate JSON metadata for a single image"""

        try:
            # Calculate MD5 hash
            file_size = image_path.stat().st_size
            md5_hash = self.calculate_md5(image_path)

            # Extract timestamp from filename if possible
            # Expected format: face_YYYYMMDD_HHMMSS_mmm_HASH.jpg
            filename = image_path.stem
            parts = filename.split('_')

            if len(parts) >= 4 and parts[0] == 'face':
                face_id = f"{parts[1]}_{parts[2]}_{parts[3]}"
                timestamp_str = f"{parts[1]} {parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}"
            else:
                # Use file modification time as fallback
                mtime = datetime.fromtimestamp(image_path.stat().st_mtime)
                face_id = mtime.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                timestamp_str = mtime.strftime("%Y-%m-%d %H:%M:%S")

            # Create basic metadata
            try:
                rel_path = str(image_path.relative_to(Path.cwd()))
            except ValueError:
                # If not relative to cwd, use absolute path
                rel_path = str(image_path)

            metadata = {
                'filename': image_path.name,
                'file_path': rel_path,
                'face_id': face_id,
                'md5_hash': md5_hash,
                'download_timestamp': datetime.now().isoformat(),
                'download_date': timestamp_str,
                'source_url': 'unknown (generated retroactively)',
                'http_status_code': 200,
                'file_size_bytes': file_size,
                'file_size_kb': round(file_size / 1024, 2),
            }

            # Add image properties
            try:
                with Image.open(image_path) as img:
                    metadata['image_properties'] = {
                        'width': img.size[0],
                        'height': img.size[1],
                        'format': img.format,
                        'mode': img.mode,
                        'dimensions': f"{img.size[0]}x{img.size[1]}"
                    }
            except Exception as e:
                self.console.print(f"[yellow]Warning:[/yellow] Could not read image properties: {e}")

            # Add face analysis if available
            if self.face_analyzer:
                try:
                    features = self.face_analyzer.analyze_face(str(image_path))
                    metadata['face_features'] = features

                    # Add queryable attributes
                    metadata['queryable_attributes'] = {
                        'brightness_level': 'bright' if features.get('brightness', 0) > 150 else 'dark',
                        'image_quality': 'high' if features.get('contrast', 0) > 50 else 'medium',
                        'has_face': features.get('faces_detected', 0) > 0,
                        'face_count': features.get('faces_detected', 0),
                        'sex': features.get('estimated_sex', 'unknown'),
                        'age_group': features.get('age_group', 'unknown'),
                        'estimated_age': features.get('estimated_age', 'unknown'),
                        'skin_tone': features.get('skin_tone', 'unknown'),
                        'skin_color': features.get('skin_color', 'unknown'),
                        'hair_color': features.get('hair_color', 'unknown')
                    }
                except Exception as e:
                    self.console.print(f"[yellow]Warning:[/yellow] Face analysis failed for {image_path.name}: {e}")
                    metadata['face_features_error'] = str(e)

            # Add generation info
            metadata['metadata_generated'] = {
                'generated_at': datetime.now().isoformat(),
                'generator': 'generate_missing_metadata.py',
                'retroactive': True
            }

            # Save JSON file
            json_path = image_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.stats['generated'] += 1
            return True

        except Exception as e:
            self.console.print(f"[red]Error[/red] processing {image_path.name}: {e}")
            self.stats['errors'] += 1
            return False

    def process_directory(self):
        """Process all images in directory"""

        self.console.print(f"\n[bold cyan]Scanning directory:[/bold cyan] {self.directory.absolute()}")

        # Find images without JSON
        images_to_process = self.find_images_without_json()

        if self.stats['total_images'] == 0:
            self.console.print("[yellow]No image files found in directory[/yellow]")
            return

        self.console.print(f"\n[bold]Found:[/bold]")
        self.console.print(f"  Total images: {self.stats['total_images']}")
        self.console.print(f"  Already have JSON: {self.stats['already_have_json']}")
        self.console.print(f"  Need metadata: {len(images_to_process)}")

        if not images_to_process:
            self.console.print("\n[green]✓ All images already have JSON metadata![/green]")
            return

        # Confirm before processing
        if not self.force and not self.auto_confirm and len(images_to_process) > 10:
            self.console.print(f"\n[yellow]About to generate {len(images_to_process)} JSON files.[/yellow]")
            try:
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    self.console.print("Cancelled.")
                    return
            except EOFError:
                # Running non-interactively, proceed anyway
                pass

        # Process images with progress bar
        self.console.print(f"\n[bold green]Generating metadata...[/bold green]\n")

        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[current]}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:

            task = progress.add_task(
                "",
                total=len(images_to_process),
                current=0
            )

            for idx, img_path in enumerate(images_to_process, 1):
                # Show current file being processed
                json_name = img_path.with_suffix('.json').name

                # Print the filenames above the progress bar
                self.console.print(f"  [cyan]📄 {img_path.name}[/cyan] → [green]📝 {json_name}[/green]")

                progress.update(task, current=idx)
                self.generate_metadata_for_image(img_path)
                progress.update(task, advance=1)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print processing summary"""

        self.console.print("\n" + "="*60)
        self.console.print("[bold green]METADATA GENERATION COMPLETE![/bold green]")
        self.console.print("="*60 + "\n")

        # Create summary table
        table = Table(show_header=False, box=None)
        table.add_column("Label", style="cyan", width=30)
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Images Found", f"{self.stats['total_images']}")
        table.add_row("Already Had JSON", f"{self.stats['already_have_json']}")
        table.add_row("✅ Metadata Generated", f"{self.stats['generated']}")
        table.add_row("❌ Errors", f"{self.stats['errors']}")

        self.console.print(table)
        self.console.print(f"\n[cyan]Directory:[/cyan] {self.directory.absolute()}")

        # Count final totals
        json_count = len(list(self.directory.glob("*.json")))
        img_count = len(list(self.directory.glob("*.jpg"))) + len(list(self.directory.glob("*.jpeg")))

        self.console.print(f"\n[bold]Final counts:[/bold]")
        self.console.print(f"  JPG files: {img_count}")
        self.console.print(f"  JSON files: {json_count}")

        if json_count == img_count:
            self.console.print("\n[green]✓ All images now have metadata![/green]")
        else:
            self.console.print(f"\n[yellow]! {img_count - json_count} images still missing metadata[/yellow]")

        self.console.print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate missing JSON metadata for face images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate metadata for images in ./faces directory
  python generate_missing_metadata.py

  # Generate metadata for specific directory
  python generate_missing_metadata.py -d faces_final

  # Regenerate all metadata (force overwrite)
  python generate_missing_metadata.py -d faces_final --force

  # Check what would be generated (dry run)
  python generate_missing_metadata.py -d faces_final --dry-run
        """
    )

    parser.add_argument('-d', '--directory', type=str, default='./faces',
                        help='Directory containing face images (default: ./faces)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Regenerate metadata even if JSON already exists')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Auto-confirm without prompting')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually doing it')

    args = parser.parse_args()

    # Create generator
    generator = MetadataGenerator(
        directory=args.directory,
        force=args.force,
        auto_confirm=args.yes
    )

    # Dry run mode
    if args.dry_run:
        generator.console.print("[yellow]DRY RUN MODE - No files will be created[/yellow]\n")
        images = generator.find_images_without_json()
        generator.console.print(f"Would generate metadata for {len(images)} images:")
        for img in images[:10]:  # Show first 10
            generator.console.print(f"  - {img.name}")
        if len(images) > 10:
            generator.console.print(f"  ... and {len(images) - 10} more")
        return

    # Process directory
    generator.process_directory()


if __name__ == "__main__":
    main()
