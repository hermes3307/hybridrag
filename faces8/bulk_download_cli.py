#!/usr/bin/env python3
"""
High-Performance CLI Face Downloader
Downloads face images in bulk with multi-threading and live status display
"""

import os
import sys
import time
import requests
import hashlib
import argparse
import threading
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Dict, Any
from queue import Queue
from collections import defaultdict

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    DownloadColumn
)
from rich.layout import Layout
from rich.text import Text

# Try to import face analysis from core.py
try:
    from core import FaceAnalyzer
    from PIL import Image
    FACE_ANALYSIS_AVAILABLE = True
except ImportError:
    FACE_ANALYSIS_AVAILABLE = False


class DownloadStats:
    """Track download statistics"""
    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = 0
        self.successes = 0
        self.duplicates = 0
        self.errors = 0
        self.total_bytes = 0
        self.total_time = 0.0
        self.start_time = time.time()
        self.error_reasons = defaultdict(int)

    def increment_attempt(self):
        with self.lock:
            self.attempts += 1

    def increment_success(self, bytes_downloaded: int, elapsed: float):
        with self.lock:
            self.successes += 1
            self.total_bytes += bytes_downloaded
            self.total_time += elapsed

    def increment_duplicate(self):
        with self.lock:
            self.duplicates += 1

    def increment_error(self, reason: str = "unknown"):
        with self.lock:
            self.errors += 1
            self.error_reasons[reason] += 1

    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'attempts': self.attempts,
                'successes': self.successes,
                'duplicates': self.duplicates,
                'errors': self.errors,
                'total_bytes': self.total_bytes,
                'total_mb': self.total_bytes / (1024 * 1024),
                'elapsed_time': elapsed,
                'avg_time_per_download': self.total_time / max(1, self.successes),
                'downloads_per_sec': self.successes / max(1, elapsed),
                'avg_speed_kbps': (self.total_bytes / 1024) / max(1, self.total_time),
                'error_reasons': dict(self.error_reasons)
            }


class BulkFaceDownloader:
    """High-performance bulk face image downloader"""

    def __init__(self,
                 output_dir: str = "faces_bulk",
                 source: str = "thispersondoesnotexist",
                 num_threads: int = 8,
                 max_downloads: int = 100,
                 timeout: int = 30,
                 generate_metadata: bool = False):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.num_threads = num_threads
        self.max_downloads = max_downloads
        self.timeout = timeout
        self.generate_metadata = generate_metadata

        self.stats = DownloadStats()
        self.downloaded_hashes: Set[str] = set()
        self.hash_lock = threading.Lock()

        self.running = True
        self.console = Console()

        # Source URLs
        self.urls = {
            'thispersondoesnotexist': 'https://thispersondoesnotexist.com/',
            '100k-faces': 'https://100k-faces.vercel.app/api/random-image'
        }

        # Initialize face analyzer if metadata is requested
        self.face_analyzer = None
        if self.generate_metadata:
            if FACE_ANALYSIS_AVAILABLE:
                self.face_analyzer = FaceAnalyzer()
            else:
                self.console.print("[yellow]Warning: Face analysis not available. Install required packages.[/yellow]")
                self.console.print("[yellow]Metadata will be basic (no face features).[/yellow]")

    def download_single_face(self) -> Optional[str]:
        """Download a single face image"""
        self.stats.increment_attempt()
        start_time = time.time()
        download_time = datetime.now()

        try:
            url = self.urls.get(self.source, self.urls['thispersondoesnotexist'])
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Calculate hash for deduplication
            image_hash = hashlib.md5(response.content).hexdigest()

            # Check for duplicates
            with self.hash_lock:
                if image_hash in self.downloaded_hashes:
                    self.stats.increment_duplicate()
                    return None
                self.downloaded_hashes.add(image_hash)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"face_{timestamp}_{image_hash[:8]}.jpg"
            file_path = self.output_dir / filename

            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Generate metadata if requested
            if self.generate_metadata:
                self._generate_metadata(file_path, filename, timestamp, image_hash,
                                       response, download_time, len(response.content))

            elapsed = time.time() - start_time
            self.stats.increment_success(len(response.content), elapsed)

            return str(file_path)

        except requests.exceptions.Timeout:
            self.stats.increment_error("timeout")
            return None
        except requests.exceptions.ConnectionError:
            self.stats.increment_error("connection")
            return None
        except requests.exceptions.HTTPError as e:
            self.stats.increment_error(f"http_{e.response.status_code}")
            return None
        except Exception as e:
            self.stats.increment_error(str(type(e).__name__))
            return None

    def _generate_metadata(self, file_path: Path, filename: str, timestamp: str,
                          image_hash: str, response, download_time: datetime,
                          file_size_bytes: int) -> None:
        """Generate and save JSON metadata for downloaded image"""
        try:
            # Basic metadata
            metadata: Dict[str, Any] = {
                'filename': filename,
                'file_path': str(file_path),
                'face_id': timestamp,
                'md5_hash': image_hash,
                'download_timestamp': download_time.isoformat(),
                'download_date': download_time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_url': self.urls.get(self.source),
                'http_status_code': response.status_code,
                'file_size_bytes': file_size_bytes,
                'file_size_kb': round(file_size_bytes / 1024, 2),
            }

            # Add image properties
            if FACE_ANALYSIS_AVAILABLE:
                with Image.open(file_path) as img:
                    metadata['image_properties'] = {
                        'width': img.size[0],
                        'height': img.size[1],
                        'format': img.format,
                        'mode': img.mode,
                        'dimensions': f"{img.size[0]}x{img.size[1]}"
                    }

            # Add face analysis features if available
            if self.face_analyzer:
                try:
                    features = self.face_analyzer.analyze_face(str(file_path))
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
                    metadata['face_features_error'] = str(e)

            # Add HTTP headers
            metadata['http_headers'] = dict(response.headers)

            # Add downloader config
            metadata['downloader_config'] = {
                'storage_dir': str(self.output_dir),
                'source': self.source,
                'threads': self.num_threads
            }

            # Save metadata JSON
            json_filename = f"face_{timestamp}_{image_hash[:8]}.json"
            json_path = self.output_dir / json_filename

            with open(json_path, 'w') as json_file:
                json.dump(metadata, json_file, indent=2, default=str)

        except Exception as e:
            # Don't fail the download if metadata generation fails
            pass

    def worker_thread(self, worker_id: int, progress: Progress, task_id: int):
        """Worker thread for downloading"""
        while self.running and self.stats.successes < self.max_downloads:
            result = self.download_single_face()
            if result:
                progress.update(task_id, advance=1)

            # Small delay to avoid hammering the server
            time.sleep(0.1)

    def generate_live_display(self, progress: Progress) -> Layout:
        """Generate the live display layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="stats", size=12),
            Layout(name="system", size=8),
        )

        # Header
        header_text = Text("🚀 HIGH-PERFORMANCE FACE DOWNLOADER", style="bold magenta", justify="center")
        layout["header"].update(Panel(header_text, style="bold blue"))

        # Progress bar
        layout["progress"].update(Panel(progress))

        # Statistics table
        stats = self.stats.get_stats()

        stats_table = Table(show_header=True, header_style="bold cyan", expand=True)
        stats_table.add_column("Metric", style="yellow", width=25)
        stats_table.add_column("Value", style="green", justify="right")

        stats_table.add_row("✅ Successful Downloads", f"{stats['successes']:,}")
        stats_table.add_row("🔄 Total Attempts", f"{stats['attempts']:,}")
        stats_table.add_row("🔁 Duplicates", f"{stats['duplicates']:,}")
        stats_table.add_row("❌ Errors", f"{stats['errors']:,}")
        stats_table.add_row("", "")
        stats_table.add_row("📦 Total Downloaded", f"{stats['total_mb']:.2f} MB")
        stats_table.add_row("⚡ Avg Speed", f"{stats['avg_speed_kbps']:.1f} KB/s")
        stats_table.add_row("⏱️  Avg Time/Download", f"{stats['avg_time_per_download']:.2f}s")
        stats_table.add_row("📊 Downloads/Second", f"{stats['downloads_per_sec']:.2f}")

        layout["stats"].update(Panel(stats_table, title="📈 Statistics", border_style="green"))

        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.output_dir))
        net_io = psutil.net_io_counters()

        system_table = Table(show_header=True, header_style="bold cyan", expand=True)
        system_table.add_column("Resource", style="yellow", width=20)
        system_table.add_column("Usage", style="cyan", justify="right")

        system_table.add_row("🔧 Active Threads", f"{threading.active_count()}/{self.num_threads + 1}")
        system_table.add_row("💻 CPU Usage", f"{cpu_percent:.1f}%")
        system_table.add_row("🧠 Memory Usage", f"{memory.percent:.1f}% ({memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f} GB)")
        system_table.add_row("💾 Disk Free", f"{disk.free / (1024**3):.1f} GB")
        system_table.add_row("📡 Network Sent", f"{net_io.bytes_sent / (1024**2):.1f} MB")
        system_table.add_row("📥 Network Recv", f"{net_io.bytes_recv / (1024**2):.1f} MB")

        layout["system"].update(Panel(system_table, title="⚙️  System Resources", border_style="blue"))

        return layout

    def start_download(self):
        """Start the bulk download process"""
        self.console.print(f"\n[bold green]Starting bulk download...[/bold green]")
        self.console.print(f"[cyan]Source:[/cyan] {self.source}")
        self.console.print(f"[cyan]Threads:[/cyan] {self.num_threads}")
        self.console.print(f"[cyan]Target:[/cyan] {self.max_downloads} downloads")
        self.console.print(f"[cyan]Output:[/cyan] {self.output_dir}\n")

        # Create progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )

        task_id = progress.add_task(
            f"[cyan]Downloading faces...",
            total=self.max_downloads
        )

        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=self.worker_thread,
                args=(i, progress, task_id),
                daemon=True
            )
            thread.start()
            threads.append(thread)

        # Live display
        try:
            with Live(self.generate_live_display(progress), console=self.console, refresh_per_second=4) as live:
                while self.running and self.stats.successes < self.max_downloads:
                    time.sleep(0.25)
                    live.update(self.generate_live_display(progress))

                # Wait for threads to finish
                self.running = False
                for thread in threads:
                    thread.join(timeout=2)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Download interrupted by user[/yellow]")
            self.running = False

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final summary"""
        stats = self.stats.get_stats()

        self.console.print("\n" + "="*60)
        self.console.print("[bold green]DOWNLOAD COMPLETE![/bold green]", justify="center")
        self.console.print("="*60 + "\n")

        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Label", style="cyan", width=30)
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("✅ Successfully Downloaded", f"{stats['successes']:,} faces")
        summary_table.add_row("📦 Total Size", f"{stats['total_mb']:.2f} MB")
        summary_table.add_row("⏱️  Total Time", f"{stats['elapsed_time']:.1f}s")
        summary_table.add_row("⚡ Average Speed", f"{stats['avg_speed_kbps']:.1f} KB/s")
        summary_table.add_row("📊 Download Rate", f"{stats['downloads_per_sec']:.2f} faces/sec")
        summary_table.add_row("🔁 Duplicates Skipped", f"{stats['duplicates']:,}")
        summary_table.add_row("❌ Errors", f"{stats['errors']:,}")

        self.console.print(summary_table)

        if stats['error_reasons']:
            self.console.print("\n[bold yellow]Error Breakdown:[/bold yellow]")
            for reason, count in stats['error_reasons'].items():
                self.console.print(f"  • {reason}: {count}")

        self.console.print(f"\n[cyan]Output directory:[/cyan] {self.output_dir.absolute()}")
        self.console.print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="High-performance CLI bulk face image downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 100 faces with 8 threads (default)
  python bulk_download_cli.py

  # Download 500 faces with 16 threads
  python bulk_download_cli.py -n 500 -t 16

  # Download with JSON metadata (includes face analysis)
  python bulk_download_cli.py -n 100 -m

  # Use 100k-faces source
  python bulk_download_cli.py -s 100k-faces -n 200

  # Custom output directory with metadata
  python bulk_download_cli.py -o my_faces -n 100 -m
        """
    )

    parser.add_argument('-n', '--num', type=int, default=100,
                        help='Number of faces to download (default: 100)')
    parser.add_argument('-t', '--threads', type=int, default=8,
                        help='Number of worker threads (default: 8)')
    parser.add_argument('-o', '--output', type=str, default='faces_bulk',
                        help='Output directory (default: faces_bulk)')
    parser.add_argument('-s', '--source', type=str,
                        choices=['thispersondoesnotexist', '100k-faces'],
                        default='thispersondoesnotexist',
                        help='Download source (default: thispersondoesnotexist)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Request timeout in seconds (default: 30)')
    parser.add_argument('-m', '--metadata', action='store_true',
                        help='Generate JSON metadata files with face analysis (slower)')

    args = parser.parse_args()

    # Auto-adjust threads based on available cores
    max_cores = psutil.cpu_count()
    if args.threads > max_cores * 2:
        print(f"[WARNING] {args.threads} threads requested but only {max_cores} CPU cores available")
        print(f"[WARNING] Recommended: {max_cores * 2} threads or less")

    downloader = BulkFaceDownloader(
        output_dir=args.output,
        source=args.source,
        num_threads=args.threads,
        max_downloads=args.num,
        timeout=args.timeout,
        generate_metadata=args.metadata
    )

    downloader.start_download()


if __name__ == "__main__":
    main()
