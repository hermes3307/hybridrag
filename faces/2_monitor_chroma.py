#!/usr/bin/env python3
"""
Real-time ChromaDB Vector Database Monitor
Beautiful graphical interface to monitor embedding status during processing
"""

import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box
    from rich.align import Align
except ImportError:
    print("Installing required package: rich")
    os.system("pip3 install rich")
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich import box
    from rich.align import Align

from face_database import FaceDatabase


class ChromaDBMonitor:
    """Real-time monitor for ChromaDB embedding process"""

    def __init__(self, refresh_rate: float = 1.0):
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.running = True
        self.face_db: Optional[FaceDatabase] = None
        self.start_time = time.time()
        self.previous_count = 0
        self.rates_history = []
        self.max_history = 10

    def initialize_database(self):
        """Initialize connection to ChromaDB"""
        try:
            self.face_db = FaceDatabase()
            return True
        except Exception as e:
            self.console.print(f"[red]❌ Failed to connect to ChromaDB: {e}[/red]")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            if not self.face_db:
                return {}

            stats = self.face_db.get_database_stats()

            # Calculate processing rate
            total_faces = stats.get('total_faces', 0)
            elapsed = time.time() - self.start_time

            # Calculate instant rate
            if elapsed > 0:
                current_rate = (total_faces - self.previous_count) / self.refresh_rate
                self.rates_history.append(current_rate)
                if len(self.rates_history) > self.max_history:
                    self.rates_history.pop(0)
                avg_rate = sum(self.rates_history) / len(self.rates_history) if self.rates_history else 0
            else:
                avg_rate = 0
                current_rate = 0

            self.previous_count = total_faces

            stats['processing_rate'] = avg_rate
            stats['current_rate'] = current_rate
            stats['elapsed_time'] = elapsed

            return stats

        except Exception as e:
            return {'error': str(e)}

    def get_collection_details(self) -> Dict[str, Any]:
        """Get detailed collection information"""
        try:
            if not self.face_db:
                return {}

            collection = self.face_db.collection
            count = collection.count()

            # Get sample metadata to analyze
            details = {
                'name': collection.name,
                'count': count,
                'metadata': collection.metadata if hasattr(collection, 'metadata') else {}
            }

            # Try to get embedding dimensions
            if count > 0:
                try:
                    result = collection.get(limit=1, include=['embeddings'])
                    if result and 'embeddings' in result and result['embeddings']:
                        details['embedding_dimension'] = len(result['embeddings'][0])
                except:
                    details['embedding_dimension'] = 'Unknown'

            return details

        except Exception as e:
            return {'error': str(e)}

    def create_header_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create header panel with title and time"""
        elapsed = stats.get('elapsed_time', 0)
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("🔍 ChromaDB Vector Database Monitor\n", style="bold cyan")
        header_text.append(f"Current Time: {current_time} | ", style="dim")
        header_text.append(f"Runtime: {time_str}", style="yellow bold")

        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="cyan"
        )

    def create_stats_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create main statistics panel"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan bold", width=30)
        table.add_column("Value", style="green bold", width=20)
        table.add_column("Info", style="yellow", width=30)

        total_faces = stats.get('total_faces', 0)
        rate = stats.get('processing_rate', 0)
        current_rate = stats.get('current_rate', 0)

        table.add_row(
            "📊 Total Embeddings",
            f"{total_faces:,}",
            "Face vectors in database"
        )

        table.add_row(
            "⚡ Average Rate",
            f"{rate:.2f} /sec",
            f"~{rate * 60:.0f} per minute"
        )

        table.add_row(
            "📈 Current Rate",
            f"{current_rate:.2f} /sec",
            "Instant processing speed"
        )

        # Estimated completion time (if we know total)
        if rate > 0:
            # Assume we're processing faces directory
            try:
                from pathlib import Path
                total_files = len(list(Path("./faces").glob("*.jpg")))
                remaining = max(0, total_files - total_faces)
                if remaining > 0 and rate > 0:
                    eta_seconds = remaining / rate
                    eta_minutes = int(eta_seconds // 60)
                    eta_hours = int(eta_minutes // 60)
                    eta_minutes = eta_minutes % 60

                    table.add_row(
                        "⏱️  Estimated Time",
                        f"{eta_hours}h {eta_minutes}m",
                        f"{remaining:,} files remaining"
                    )
            except:
                pass

        return Panel(
            table,
            title="[bold white]📈 Processing Statistics[/bold white]",
            border_style="green",
            box=box.ROUNDED
        )

    def create_distribution_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create distribution statistics panel"""
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Distribution", style="green", width=50)

        # Age groups
        age_dist = stats.get('age_group_distribution', {})
        if age_dist:
            age_text = ", ".join([f"{k}: {v}" for k, v in sorted(age_dist.items())])
            table.add_row("👥 Age Groups", age_text)

        # Skin tones
        skin_dist = stats.get('skin_tone_distribution', {})
        if skin_dist:
            skin_text = ", ".join([f"{k}: {v}" for k, v in sorted(skin_dist.items())])
            table.add_row("🎨 Skin Tones", skin_text)

        # Quality
        quality_dist = stats.get('quality_distribution', {})
        if quality_dist:
            quality_text = ", ".join([f"{k}: {v}" for k, v in sorted(quality_dist.items())])
            table.add_row("⭐ Quality", quality_text)

        if not age_dist and not skin_dist and not quality_dist:
            table.add_row("No Data", "Waiting for embeddings with metadata...")

        return Panel(
            table,
            title="[bold white]📊 Data Distribution[/bold white]",
            border_style="magenta",
            box=box.ROUNDED
        )

    def create_collection_panel(self, details: Dict[str, Any]) -> Panel:
        """Create collection details panel"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Property", style="cyan bold", width=25)
        table.add_column("Value", style="yellow", width=55)

        table.add_row("Collection Name", details.get('name', 'N/A'))
        table.add_row("Document Count", f"{details.get('count', 0):,}")
        table.add_row("Embedding Dimension", str(details.get('embedding_dimension', 'N/A')))

        # Database location
        if self.face_db:
            table.add_row("Database Path", "./chroma_db")

        return Panel(
            table,
            title="[bold white]🗄️  Collection Info[/bold white]",
            border_style="blue",
            box=box.ROUNDED
        )

    def create_progress_panel(self, stats: Dict[str, Any]) -> Panel:
        """Create visual progress panel"""
        total_faces = stats.get('total_faces', 0)

        # Create visual progress bar
        try:
            from pathlib import Path
            total_files = len(list(Path("./faces").glob("*.jpg")))
            progress_pct = min(100, (total_faces / total_files * 100) if total_files > 0 else 0)

            # Create ASCII progress bar
            bar_width = 50
            filled = int(bar_width * progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            progress_text = Text()
            progress_text.append(f"\n{bar}\n\n", style="green bold")
            progress_text.append(f"{progress_pct:.1f}% Complete ", style="cyan bold")
            progress_text.append(f"({total_faces:,} / {total_files:,})", style="yellow")

        except:
            progress_text = Text(f"\n{total_faces:,} embeddings created\n", style="green bold")

        return Panel(
            Align.center(progress_text),
            title="[bold white]📊 Overall Progress[/bold white]",
            border_style="yellow",
            box=box.ROUNDED
        )

    def create_layout(self) -> Layout:
        """Create the main layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="stats"),
            Layout(name="progress")
        )

        layout["right"].split_column(
            Layout(name="collection"),
            Layout(name="distribution")
        )

        return layout

    def create_footer_panel(self) -> Panel:
        """Create footer with controls"""
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to exit | Refresh rate: ", style="dim")
        footer_text.append(f"{self.refresh_rate}s", style="bold yellow")
        footer_text.append(" | Database: ", style="dim")
        footer_text.append("ChromaDB", style="bold green")

        return Panel(
            Align.center(footer_text),
            style="dim"
        )

    def generate_display(self) -> Layout:
        """Generate the complete display layout"""
        stats = self.get_database_stats()
        details = self.get_collection_details()

        layout = self.create_layout()

        layout["header"].update(self.create_header_panel(stats))
        layout["stats"].update(self.create_stats_panel(stats))
        layout["progress"].update(self.create_progress_panel(stats))
        layout["collection"].update(self.create_collection_panel(details))
        layout["distribution"].update(self.create_distribution_panel(stats))
        layout["footer"].update(self.create_footer_panel())

        return layout

    def start_monitoring(self):
        """Start the monitoring display"""
        self.console.clear()

        if not self.initialize_database():
            return

        self.console.print("\n[bold green]✨ ChromaDB Monitor Started[/bold green]\n")
        self.console.print("[yellow]Connecting to database...[/yellow]\n")

        try:
            with Live(self.generate_display(), refresh_per_second=1/self.refresh_rate, console=self.console) as live:
                while self.running:
                    try:
                        live.update(self.generate_display())
                        time.sleep(self.refresh_rate)
                    except KeyboardInterrupt:
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print("\n\n[bold cyan]👋 Monitoring stopped[/bold cyan]\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time ChromaDB Vector Database Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 monitor_chroma.py                    # Monitor with 1 second refresh
  python3 monitor_chroma.py --refresh 0.5      # Monitor with 0.5 second refresh
  python3 monitor_chroma.py --refresh 2        # Monitor with 2 second refresh
        """
    )

    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Refresh rate in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    # Validate refresh rate
    if args.refresh < 0.1:
        print("⚠️  Minimum refresh rate is 0.1 seconds")
        args.refresh = 0.1
    elif args.refresh > 10:
        print("⚠️  Maximum refresh rate is 10 seconds")
        args.refresh = 10

    # Create and start monitor
    monitor = ChromaDBMonitor(refresh_rate=args.refresh)

    try:
        monitor.start_monitoring()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
