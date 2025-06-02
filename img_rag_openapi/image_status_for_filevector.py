#!/usr/bin/env python3
"""
📊 Vector Database Status Monitor
벡터 데이터베이스 상태 모니터링 및 대시보드 시스템
"""

import os
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Rich library for beautiful console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("💡 Install 'rich' for beautiful dashboard: pip install rich")

logger = logging.getLogger(__name__)

@dataclass
class StorageStats:
    """저장소 통계 데이터 클래스"""
    total_images: int = 0
    total_vectors: int = 0
    total_metadata: int = 0
    total_size_bytes: int = 0
    vector_size_bytes: int = 0
    metadata_size_bytes: int = 0
    db_size_bytes: int = 0
    vector_dimension: int = 512
    collection_name: str = ""
    storage_type: str = ""
    last_updated: str = ""
    processing_stats: Dict = None
    error_count: int = 0
    warning_count: int = 0

@dataclass
class ImageStats:
    """이미지 통계 데이터 클래스"""
    by_format: Dict[str, int] = None
    by_resolution: Dict[str, int] = None
    by_size_range: Dict[str, int] = None
    avg_sharpness: float = 0.0
    avg_brightness: float = 0.0
    avg_filesize: float = 0.0
    dominant_colors_distribution: Dict = None

class VectorDatabaseStatusMonitor:
    """📊 벡터 데이터베이스 상태 모니터링 시스템"""
    
    def __init__(self, storage_path: str = "vector_storage", 
                 collection_name: str = "image_vectors"):
        """Initialize status monitor"""
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.collection_path = self.storage_path / collection_name
        
        # Initialize console for rich output
        self.console = Console() if HAS_RICH else None
        
        # Cache for performance
        self._stats_cache = {}
        self._cache_timeout = 30  # seconds
        self._last_cache_time = 0
        
        logger.info(f"📊 Status monitor initialized for {self.collection_path}")
    
    def get_storage_stats(self, force_refresh: bool = False) -> StorageStats:
        """전체 저장소 통계 수집"""
        # Check cache
        current_time = time.time()
        if not force_refresh and current_time - self._last_cache_time < self._cache_timeout:
            if 'storage_stats' in self._stats_cache:
                return self._stats_cache['storage_stats']
        
        try:
            stats = StorageStats()
            
            if not self.collection_path.exists():
                logger.warning(f"Collection path does not exist: {self.collection_path}")
                return stats
            
            stats.collection_name = self.collection_name
            stats.storage_type = "file_based"
            stats.last_updated = datetime.now().isoformat()
            
            # Vector files statistics
            vector_dir = self.collection_path / "vectors"
            if vector_dir.exists():
                vector_files = list(vector_dir.glob("*.npy"))
                stats.total_vectors = len(vector_files)
                stats.vector_size_bytes = sum(f.stat().st_size for f in vector_files)
            
            # Metadata files statistics
            metadata_dir = self.collection_path / "metadata"
            if metadata_dir.exists():
                metadata_files = list(metadata_dir.glob("*.json"))
                stats.total_metadata = len(metadata_files)
                stats.metadata_size_bytes = sum(f.stat().st_size for f in metadata_files)
            
            # Database statistics
            db_path = self.collection_path / "metadata.db"
            if db_path.exists():
                stats.db_size_bytes = db_path.stat().st_size
                stats.total_images = self._count_images_in_db()
            
            # Total size
            stats.total_size_bytes = (stats.vector_size_bytes + 
                                    stats.metadata_size_bytes + 
                                    stats.db_size_bytes)
            
            # Processing statistics
            stats.processing_stats = self._get_processing_stats()
            
            # Update cache
            self._stats_cache['storage_stats'] = stats
            self._last_cache_time = current_time
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting storage stats: {e}")
            return StorageStats()
    
    def get_image_stats(self) -> ImageStats:
        """이미지별 상세 통계 수집"""
        try:
            stats = ImageStats()
            
            db_path = self.collection_path / "metadata.db"
            if not db_path.exists():
                return stats
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Format distribution
            cursor.execute("SELECT format, COUNT(*) FROM image_metadata GROUP BY format")
            stats.by_format = dict(cursor.fetchall())
            
            # Resolution categories
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN width >= 3840 THEN '4K+'
                        WHEN width >= 1920 THEN 'FHD'
                        WHEN width >= 1280 THEN 'HD'
                        WHEN width >= 720 THEN 'SD'
                        ELSE 'Small'
                    END as res_category,
                    COUNT(*)
                FROM image_metadata 
                GROUP BY res_category
            """)
            stats.by_resolution = dict(cursor.fetchall())
            
            # File size ranges
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN file_size >= 10485760 THEN '10MB+'
                        WHEN file_size >= 5242880 THEN '5-10MB'
                        WHEN file_size >= 1048576 THEN '1-5MB'
                        WHEN file_size >= 102400 THEN '100KB-1MB'
                        ELSE '<100KB'
                    END as size_range,
                    COUNT(*)
                FROM image_metadata 
                GROUP BY size_range
            """)
            stats.by_size_range = dict(cursor.fetchall())
            
            # Average values
            cursor.execute("""
                SELECT 
                    AVG(sharpness), AVG(brightness), AVG(file_size)
                FROM image_metadata
            """)
            avg_data = cursor.fetchone()
            if avg_data and avg_data[0] is not None:
                stats.avg_sharpness = float(avg_data[0])
                stats.avg_brightness = float(avg_data[1])
                stats.avg_filesize = float(avg_data[2])
            
            # Dominant colors analysis
            stats.dominant_colors_distribution = self._analyze_color_distribution()
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting image stats: {e}")
            return ImageStats()
    
    def _count_images_in_db(self) -> int:
        """데이터베이스의 이미지 수 카운트"""
        try:
            db_path = self.collection_path / "metadata.db"
            if not db_path.exists():
                return 0
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM image_metadata")
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            return 0
    
    def _get_processing_stats(self) -> Dict:
        """처리 통계 수집"""
        try:
            # Processing log analysis (if exists)
            processing_stats = {
                'last_processed': None,
                'processing_rate': 0,
                'failed_count': 0,
                'success_rate': 0,
                'recent_activity': []
            }
            
            # Check recent metadata files for processing timeline
            metadata_dir = self.collection_path / "metadata"
            if metadata_dir.exists():
                metadata_files = sorted(
                    metadata_dir.glob("*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if metadata_files:
                    # Most recent processing
                    latest_file = metadata_files[0]
                    processing_stats['last_processed'] = datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).isoformat()
                    
                    # Recent activity (last 24 hours)
                    yesterday = time.time() - 86400
                    recent_files = [
                        f for f in metadata_files 
                        if f.stat().st_mtime > yesterday
                    ]
                    processing_stats['recent_activity'] = len(recent_files)
            
            return processing_stats
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
    
    def _analyze_color_distribution(self) -> Dict:
        """색상 분포 분석"""
        try:
            color_analysis = {
                'most_common_hues': {},
                'brightness_distribution': {},
                'saturation_levels': {}
            }
            
            db_path = self.collection_path / "metadata.db"
            if not db_path.exists():
                return color_analysis
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Brightness distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN brightness >= 200 THEN 'Very Bright'
                        WHEN brightness >= 150 THEN 'Bright'
                        WHEN brightness >= 100 THEN 'Normal'
                        WHEN brightness >= 50 THEN 'Dark'
                        ELSE 'Very Dark'
                    END as brightness_level,
                    COUNT(*)
                FROM image_metadata 
                WHERE brightness IS NOT NULL
                GROUP BY brightness_level
            """)
            color_analysis['brightness_distribution'] = dict(cursor.fetchall())
            
            # Saturation levels
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN saturation >= 80 THEN 'Very Saturated'
                        WHEN saturation >= 60 THEN 'Saturated'
                        WHEN saturation >= 40 THEN 'Moderate'
                        WHEN saturation >= 20 THEN 'Desaturated'
                        ELSE 'Grayscale'
                    END as saturation_level,
                    COUNT(*)
                FROM image_metadata 
                WHERE saturation IS NOT NULL
                GROUP BY saturation_level
            """)
            color_analysis['saturation_levels'] = dict(cursor.fetchall())
            
            conn.close()
            return color_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {}
    
    def get_health_status(self) -> Dict:
        """시스템 건강 상태 체크"""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            stats = self.get_storage_stats()
            
            # Check for issues
            if stats.total_vectors != stats.total_metadata:
                health['issues'].append(
                    f"Vector-Metadata mismatch: {stats.total_vectors} vectors vs {stats.total_metadata} metadata"
                )
                health['status'] = 'warning'
            
            if stats.total_size_bytes > 10 * 1024 * 1024 * 1024:  # 10GB
                health['warnings'].append(
                    f"Large storage size: {stats.total_size_bytes / 1024**3:.1f}GB"
                )
            
            if stats.total_images < 10:
                health['recommendations'].append(
                    "Consider adding more images for better search results"
                )
            
            # Check database integrity
            if not self._check_database_integrity():
                health['issues'].append("Database integrity check failed")
                health['status'] = 'error'
            
            if health['issues']:
                health['status'] = 'error'
            elif health['warnings']:
                health['status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                'status': 'error',
                'issues': [f"Health check failed: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }
    
    def _check_database_integrity(self) -> bool:
        """데이터베이스 무결성 체크"""
        try:
            db_path = self.collection_path / "metadata.db"
            if not db_path.exists():
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Simple integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            return result[0] == 'ok'
            
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return False
    
    def format_size(self, size_bytes: int) -> str:
        """바이트를 읽기 쉬운 형태로 변환"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def print_status_simple(self) -> None:
        """간단한 텍스트 상태 출력"""
        storage_stats = self.get_storage_stats()
        image_stats = self.get_image_stats()
        health = self.get_health_status()
        
        print("=" * 60)
        print("📊 VECTOR DATABASE STATUS")
        print("=" * 60)
        
        print(f"\n📁 Collection: {storage_stats.collection_name}")
        print(f"🏪 Storage Type: {storage_stats.storage_type}")
        print(f"📍 Path: {self.collection_path}")
        print(f"🕐 Last Updated: {storage_stats.last_updated}")
        print(f"❤️  Health: {health['status'].upper()}")
        
        print(f"\n📈 STORAGE STATISTICS")
        print(f"  🖼️  Total Images: {storage_stats.total_images:,}")
        print(f"  🎯 Total Vectors: {storage_stats.total_vectors:,}")
        print(f"  📋 Total Metadata: {storage_stats.total_metadata:,}")
        print(f"  💾 Total Size: {self.format_size(storage_stats.total_size_bytes)}")
        print(f"     ├─ Vectors: {self.format_size(storage_stats.vector_size_bytes)}")
        print(f"     ├─ Metadata: {self.format_size(storage_stats.metadata_size_bytes)}")
        print(f"     └─ Database: {self.format_size(storage_stats.db_size_bytes)}")
        
        if image_stats.by_format:
            print(f"\n🎨 IMAGE BREAKDOWN")
            print(f"  📊 By Format:")
            for fmt, count in image_stats.by_format.items():
                print(f"     ├─ {fmt}: {count:,}")
            
            if image_stats.by_resolution:
                print(f"  📐 By Resolution:")
                for res, count in image_stats.by_resolution.items():
                    print(f"     ├─ {res}: {count:,}")
        
        if image_stats.avg_sharpness > 0:
            print(f"\n📊 QUALITY METRICS")
            print(f"  ✨ Avg Sharpness: {image_stats.avg_sharpness:.1f}")
            print(f"  ☀️  Avg Brightness: {image_stats.avg_brightness:.1f}")
            print(f"  📦 Avg File Size: {self.format_size(image_stats.avg_filesize)}")
        
        if health['issues'] or health['warnings']:
            print(f"\n⚠️  HEALTH ISSUES")
            for issue in health['issues']:
                print(f"  ❌ {issue}")
            for warning in health['warnings']:
                print(f"  ⚠️  {warning}")
        
        if health['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in health['recommendations']:
                print(f"  💡 {rec}")
        
        print("=" * 60)
    
    def print_status_rich(self) -> None:
        """Rich 라이브러리를 사용한 아름다운 대시보드"""
        if not HAS_RICH:
            self.print_status_simple()
            return
        
        storage_stats = self.get_storage_stats()
        image_stats = self.get_image_stats()
        health = self.get_health_status()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="storage", size=12),
            Layout(name="images")
        )
        
        layout["right"].split_column(
            Layout(name="quality", size=10),
            Layout(name="health")
        )
        
        # Header
        header_text = Text("📊 VECTOR DATABASE STATUS DASHBOARD", 
                          justify="center", style="bold blue")
        layout["header"].update(Panel(header_text, border_style="blue"))
        
        # Storage Statistics
        storage_table = Table(title="📁 Storage Statistics", border_style="green")
        storage_table.add_column("Metric", style="cyan", no_wrap=True)
        storage_table.add_column("Value", style="magenta")
        
        storage_table.add_row("Collection", storage_stats.collection_name)
        storage_table.add_row("Storage Type", storage_stats.storage_type)
        storage_table.add_row("Total Images", f"{storage_stats.total_images:,}")
        storage_table.add_row("Total Vectors", f"{storage_stats.total_vectors:,}")
        storage_table.add_row("Total Metadata", f"{storage_stats.total_metadata:,}")
        storage_table.add_row("Total Size", self.format_size(storage_stats.total_size_bytes))
        storage_table.add_row("Vector Size", self.format_size(storage_stats.vector_size_bytes))
        storage_table.add_row("Metadata Size", self.format_size(storage_stats.metadata_size_bytes))
        storage_table.add_row("Database Size", self.format_size(storage_stats.db_size_bytes))
        
        layout["storage"].update(Panel(storage_table, border_style="green"))
        
        # Image Breakdown
        if image_stats.by_format:
            image_table = Table(title="🎨 Image Breakdown", border_style="yellow")
            image_table.add_column("Category", style="cyan")
            image_table.add_column("Type", style="white")
            image_table.add_column("Count", style="magenta")
            
            for fmt, count in image_stats.by_format.items():
                image_table.add_row("Format", fmt, f"{count:,}")
            
            if image_stats.by_resolution:
                for res, count in image_stats.by_resolution.items():
                    image_table.add_row("Resolution", res, f"{count:,}")
            
            layout["images"].update(Panel(image_table, border_style="yellow"))
        
        # Quality Metrics
        if image_stats.avg_sharpness > 0:
            quality_table = Table(title="📊 Quality Metrics", border_style="cyan")
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Average", style="magenta")
            
            quality_table.add_row("Sharpness", f"{image_stats.avg_sharpness:.1f}")
            quality_table.add_row("Brightness", f"{image_stats.avg_brightness:.1f}")
            quality_table.add_row("File Size", self.format_size(image_stats.avg_filesize))
            
            layout["quality"].update(Panel(quality_table, border_style="cyan"))
        
        # Health Status
        health_color = {"healthy": "green", "warning": "yellow", "error": "red"}
        health_text = Text(f"Status: {health['status'].upper()}", 
                          style=f"bold {health_color.get(health['status'], 'white')}")
        
        health_content = [health_text]
        
        if health['issues']:
            health_content.append(Text("\n❌ Issues:", style="bold red"))
            for issue in health['issues']:
                health_content.append(Text(f"  • {issue}", style="red"))
        
        if health['warnings']:
            health_content.append(Text("\n⚠️  Warnings:", style="bold yellow"))
            for warning in health['warnings']:
                health_content.append(Text(f"  • {warning}", style="yellow"))
        
        health_panel = Panel(
            Text.assemble(*health_content),
            title="❤️  Health Status",
            border_style=health_color.get(health['status'], 'white')
        )
        layout["health"].update(health_panel)
        
        # Footer
        footer_text = Text(
            f"Last Updated: {storage_stats.last_updated} | "
            f"Path: {self.collection_path}",
            justify="center", style="dim"
        )
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        # Print the dashboard
        self.console.print(layout)
    
    def print_status(self, rich_output: bool = True) -> None:
        """상태 출력 (Rich 사용 여부 선택)"""
        if rich_output and HAS_RICH:
            self.print_status_rich()
        else:
            self.print_status_simple()
    
    def export_status_json(self, output_path: str = None) -> str:
        """상태를 JSON 파일로 출력"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'storage_stats': self.get_storage_stats().__dict__,
                'image_stats': self.get_image_stats().__dict__,
                'health_status': self.get_health_status()
            }
            
            if output_path is None:
                output_path = f"vector_db_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            logger.info(f"Status exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting status: {e}")
            return ""
    
    def watch_status(self, interval: int = 5) -> None:
        """실시간 상태 모니터링"""
        if not HAS_RICH:
            print("Real-time monitoring requires 'rich' library: pip install rich")
            return
        
        def generate_display():
            while True:
                # Force refresh stats
                self.get_storage_stats(force_refresh=True)
                
                # Create simplified live display
                table = Table(title="📊 Live Vector Database Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                stats = self.get_storage_stats()
                health = self.get_health_status()
                
                table.add_row("Images", f"{stats.total_images:,}")
                table.add_row("Vectors", f"{stats.total_vectors:,}")
                table.add_row("Total Size", self.format_size(stats.total_size_bytes))
                table.add_row("Health", health['status'].upper())
                table.add_row("Last Update", stats.last_updated.split('T')[1][:8])
                
                yield Panel(table, title="🔄 Live Monitor", border_style="blue")
                time.sleep(interval)
        
        try:
            with Live(generate_display(), refresh_per_second=1/interval) as live:
                while True:
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.console.print("\n👋 Monitoring stopped")

# CLI Interface
def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Database Status Monitor")
    parser.add_argument("--storage-path", default="vector_storage", 
                       help="Path to vector storage")
    parser.add_argument("--collection", default="image_vectors", 
                       help="Collection name")
    parser.add_argument("--export", help="Export status to JSON file")
    parser.add_argument("--watch", action="store_true", 
                       help="Enable live monitoring")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Watch interval in seconds")
    parser.add_argument("--simple", action="store_true", 
                       help="Use simple text output")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = VectorDatabaseStatusMonitor(args.storage_path, args.collection)
    
    if args.watch:
        print("🔄 Starting live monitoring... (Press Ctrl+C to stop)")
        monitor.watch_status(args.interval)
    elif args.export:
        output_file = monitor.export_status_json(args.export)
        print(f"📄 Status exported to {output_file}")
    else:
        # Print status
        monitor.print_status(rich_output=not args.simple)

if __name__ == "__main__":
    main()