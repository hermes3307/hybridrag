#!/usr/bin/env python3
"""
📊 Qdrant Vector Database Status Monitor
Qdrant 서버 상태 모니터링 및 대시보드 시스템
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
import asyncio

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

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
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)

@dataclass
class QdrantCollectionStats:
    """Qdrant 컬렉션 통계"""
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    disk_data_size: int
    ram_data_size: int
    config: Dict
    status: str
    optimizer_status: Dict
    payload_schema: Dict

@dataclass
class QdrantServerStats:
    """Qdrant 서버 통계"""
    collections: List[QdrantCollectionStats]
    total_collections: int
    total_vectors: int
    total_points: int
    total_disk_size: int
    total_ram_size: int
    server_version: str
    uptime: Optional[str]
    last_updated: str

class QdrantStatusMonitor:
    """📊 Qdrant 벡터 데이터베이스 상태 모니터링"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 api_key: Optional[str] = None, timeout: int = 30):
        """Initialize Qdrant status monitor"""
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(
                host=host, 
                port=port, 
                api_key=api_key,
                timeout=timeout
            )
            # Test connection
            self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize console for rich output
        self.console = Console() if HAS_RICH else None
        
        # Cache for performance
        self._stats_cache = {}
        self._cache_timeout = 10  # seconds
        self._last_cache_time = 0
    
    def test_connection(self) -> bool:
        """Qdrant 서버 연결 테스트"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_server_info(self) -> Dict:
        """Qdrant 서버 정보 조회"""
        try:
            # Get telemetry info (contains version and stats)
            telemetry = self.client.http.cluster_api.cluster_status()
            
            return {
                'version': getattr(telemetry, 'version', 'unknown'),
                'status': 'connected',
                'host': self.host,
                'port': self.port,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting server info: {e}")
            return {
                'version': 'unknown',
                'status': 'error',
                'host': self.host,
                'port': self.port,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_collection_stats(self, collection_name: str) -> Optional[QdrantCollectionStats]:
        """특정 컬렉션의 상세 통계"""
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            # Get collection cluster info for more detailed stats
            try:
                cluster_info = self.client.http.collections_api.get_collection_cluster_info(collection_name)
                points_count = getattr(cluster_info, 'points_count', 0)
                segments_count = len(getattr(cluster_info, 'local_shards', []))
            except:
                points_count = collection_info.points_count
                segments_count = 0
            
            return QdrantCollectionStats(
                name=collection_name,
                vectors_count=collection_info.vectors_count or 0,
                indexed_vectors_count=collection_info.indexed_vectors_count or 0,
                points_count=points_count,
                segments_count=segments_count,
                disk_data_size=0,  # Not easily available in basic API
                ram_data_size=0,   # Not easily available in basic API
                config=collection_info.config.dict() if collection_info.config else {},
                status=collection_info.status.value if collection_info.status else 'unknown',
                optimizer_status=collection_info.optimizer_status.dict() if collection_info.optimizer_status else {},
                payload_schema={}  # Would need separate API call
            )
            
        except Exception as e:
            logger.error(f"Error getting collection stats for {collection_name}: {e}")
            return None
    
    def get_all_collections_stats(self) -> QdrantServerStats:
        """모든 컬렉션의 통계 수집"""
        try:
            # Get all collections
            collections_response = self.client.get_collections()
            collections = collections_response.collections
            
            collection_stats = []
            total_vectors = 0
            total_points = 0
            
            for collection in collections:
                stats = self.get_collection_stats(collection.name)
                if stats:
                    collection_stats.append(stats)
                    total_vectors += stats.vectors_count
                    total_points += stats.points_count
            
            # Get server info
            server_info = self.get_server_info()
            
            return QdrantServerStats(
                collections=collection_stats,
                total_collections=len(collection_stats),
                total_vectors=total_vectors,
                total_points=total_points,
                total_disk_size=0,  # Sum of all collection disk sizes
                total_ram_size=0,   # Sum of all collection RAM sizes
                server_version=server_info.get('version', 'unknown'),
                uptime=None,        # Not easily available
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting server stats: {e}")
            return QdrantServerStats(
                collections=[],
                total_collections=0,
                total_vectors=0,
                total_points=0,
                total_disk_size=0,
                total_ram_size=0,
                server_version='error',
                uptime=None,
                last_updated=datetime.now().isoformat()
            )
    
    def get_collection_sample_data(self, collection_name: str, limit: int = 10) -> List[Dict]:
        """컬렉션의 샘플 데이터 조회"""
        try:
            # Get sample points
            points = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't include vectors for performance
            )
            
            sample_data = []
            for point in points[0]:  # points[0] contains the actual points
                sample_data.append({
                    'id': str(point.id),
                    'payload': point.payload,
                    'has_vector': point.vector is not None
                })
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error getting sample data from {collection_name}: {e}")
            return []
    
    def analyze_collection_content(self, collection_name: str) -> Dict:
        """컬렉션 내용 분석"""
        try:
            # Get sample data
            sample_data = self.get_collection_sample_data(collection_name, 100)
            
            if not sample_data:
                return {}
            
            # Analyze payload fields
            field_analysis = defaultdict(lambda: {'count': 0, 'types': set(), 'sample_values': []})
            
            for item in sample_data:
                payload = item.get('payload', {})
                for field, value in payload.items():
                    field_analysis[field]['count'] += 1
                    field_analysis[field]['types'].add(type(value).__name__)
                    if len(field_analysis[field]['sample_values']) < 3:
                        field_analysis[field]['sample_values'].append(value)
            
            # Convert sets to lists for JSON serialization
            for field in field_analysis:
                field_analysis[field]['types'] = list(field_analysis[field]['types'])
            
            # Analyze common payload patterns
            analysis = {
                'total_samples': len(sample_data),
                'payload_fields': dict(field_analysis),
                'common_fields': [],
                'field_coverage': {}
            }
            
            # Find most common fields
            total_samples = len(sample_data)
            for field, data in field_analysis.items():
                coverage = data['count'] / total_samples * 100
                analysis['field_coverage'][field] = coverage
                if coverage > 80:  # Fields present in >80% of samples
                    analysis['common_fields'].append(field)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing collection content: {e}")
            return {}
    
    def get_health_status(self) -> Dict:
        """Qdrant 서버 건강 상태 체크"""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Test basic connectivity
            if not self.test_connection():
                health['status'] = 'error'
                health['issues'].append('Cannot connect to Qdrant server')
                return health
            
            # Get server stats
            server_stats = self.get_all_collections_stats()
            
            # Check for issues
            if server_stats.total_collections == 0:
                health['warnings'].append('No collections found')
                health['recommendations'].append('Create collections and add some vectors')
            
            if server_stats.total_vectors == 0:
                health['warnings'].append('No vectors found in any collection')
                health['recommendations'].append('Add vectors to your collections')
            
            # Check individual collections
            for collection in server_stats.collections:
                if collection.status != 'green':
                    health['issues'].append(f'Collection {collection.name} status: {collection.status}')
                    health['status'] = 'warning'
                
                if collection.vectors_count != collection.indexed_vectors_count:
                    unindexed = collection.vectors_count - collection.indexed_vectors_count
                    health['warnings'].append(
                        f'Collection {collection.name} has {unindexed} unindexed vectors'
                    )
            
            # Performance recommendations
            if server_stats.total_vectors > 100000:
                health['recommendations'].append('Consider optimizing collections for large datasets')
            
            if health['issues']:
                health['status'] = 'error'
            elif health['warnings']:
                health['status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {
                'status': 'error',
                'issues': [f'Health check failed: {str(e)}'],
                'warnings': [],
                'recommendations': []
            }
    
    def format_size(self, size_bytes: int) -> str:
        """바이트를 읽기 쉬운 형태로 변환"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def print_status_simple(self) -> None:
        """간단한 텍스트 상태 출력"""
        try:
            server_stats = self.get_all_collections_stats()
            health = self.get_health_status()
            server_info = self.get_server_info()
            
            print("=" * 70)
            print("📊 QDRANT VECTOR DATABASE STATUS")
            print("=" * 70)
            
            print(f"\n🖥️  Server Information:")
            print(f"  📍 Host: {self.host}:{self.port}")
            print(f"  🔗 Status: {server_info['status']}")
            print(f"  📦 Version: {server_info['version']}")
            print(f"  🕐 Last Updated: {server_stats.last_updated}")
            print(f"  ❤️  Health: {health['status'].upper()}")
            
            print(f"\n📈 GLOBAL STATISTICS:")
            print(f"  📁 Total Collections: {server_stats.total_collections}")
            print(f"  🎯 Total Vectors: {server_stats.total_vectors:,}")
            print(f"  📋 Total Points: {server_stats.total_points:,}")
            
            if server_stats.collections:
                print(f"\n📚 COLLECTIONS:")
                for collection in server_stats.collections:
                    status_icon = "🟢" if collection.status == "green" else "🟡" if collection.status == "yellow" else "🔴"
                    print(f"  {status_icon} {collection.name}:")
                    print(f"     ├─ Vectors: {collection.vectors_count:,}")
                    print(f"     ├─ Indexed: {collection.indexed_vectors_count:,}")
                    print(f"     ├─ Points: {collection.points_count:,}")
                    print(f"     └─ Status: {collection.status}")
                    
                    # Show vector config
                    if collection.config.get('params', {}).get('vectors'):
                        vector_config = collection.config['params']['vectors']
                        if isinstance(vector_config, dict):
                            size = vector_config.get('size', 'unknown')
                            distance = vector_config.get('distance', 'unknown')
                            print(f"        └─ Config: {size}D, {distance}")
            
            if health['issues'] or health['warnings']:
                print(f"\n⚠️  HEALTH ISSUES:")
                for issue in health['issues']:
                    print(f"  ❌ {issue}")
                for warning in health['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if health['recommendations']:
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in health['recommendations']:
                    print(f"  💡 {rec}")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error displaying status: {e}")
    
    def print_status_rich(self) -> None:
        """Rich 라이브러리를 사용한 아름다운 대시보드"""
        if not HAS_RICH:
            self.print_status_simple()
            return
        
        try:
            server_stats = self.get_all_collections_stats()
            health = self.get_health_status()
            server_info = self.get_server_info()
            
            # Create main layout
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            
            layout["main"].split_row(
                Layout(name="left"),
                Layout(name="right")
            )
            
            layout["left"].split_column(
                Layout(name="server", size=8),
                Layout(name="collections")
            )
            
            layout["right"].split_column(
                Layout(name="stats", size=8),
                Layout(name="health")
            )
            
            # Header
            header_text = Text(
                f"🐳 QDRANT DATABASE DASHBOARD - {self.host}:{self.port}", 
                justify="center", 
                style="bold blue"
            )
            layout["header"].update(Panel(header_text, border_style="blue"))
            
            # Server info
            server_table = Table(title="🖥️  Server Information", border_style="green")
            server_table.add_column("Property", style="cyan", no_wrap=True)
            server_table.add_column("Value", style="magenta")
            
            server_table.add_row("Host", f"{self.host}:{self.port}")
            server_table.add_row("Version", server_info['version'])
            server_table.add_row("Status", server_info['status'])
            server_table.add_row("Last Check", server_stats.last_updated.split('T')[1][:8])
            
            layout["server"].update(Panel(server_table, border_style="green"))
            
            # Global stats
            stats_table = Table(title="📊 Global Statistics", border_style="cyan")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Count", style="magenta")
            
            stats_table.add_row("Collections", f"{server_stats.total_collections}")
            stats_table.add_row("Total Vectors", f"{server_stats.total_vectors:,}")
            stats_table.add_row("Total Points", f"{server_stats.total_points:,}")
            
            layout["stats"].update(Panel(stats_table, border_style="cyan"))
            
            # Collections
            if server_stats.collections:
                collections_table = Table(title="📚 Collections", border_style="yellow")
                collections_table.add_column("Name", style="cyan")
                collections_table.add_column("Vectors", style="magenta")
                collections_table.add_column("Status", style="green")
                
                for collection in server_stats.collections:
                    status_style = "green" if collection.status == "green" else "yellow" if collection.status == "yellow" else "red"
                    collections_table.add_row(
                        collection.name,
                        f"{collection.vectors_count:,}",
                        Text(collection.status, style=status_style)
                    )
                
                layout["collections"].update(Panel(collections_table, border_style="yellow"))
            
            # Health status
            health_color = {"healthy": "green", "warning": "yellow", "error": "red"}
            health_text = Text(
                f"Status: {health['status'].upper()}", 
                style=f"bold {health_color.get(health['status'], 'white')}"
            )
            
            health_content = [health_text]
            
            if health['issues']:
                health_content.append(Text("\n❌ Issues:", style="bold red"))
                for issue in health['issues'][:3]:  # Show max 3 issues
                    health_content.append(Text(f"  • {issue}", style="red"))
            
            if health['warnings']:
                health_content.append(Text("\n⚠️  Warnings:", style="bold yellow"))
                for warning in health['warnings'][:3]:  # Show max 3 warnings
                    health_content.append(Text(f"  • {warning}", style="yellow"))
            
            health_panel = Panel(
                Text.assemble(*health_content),
                title="❤️  Health Status",
                border_style=health_color.get(health['status'], 'white')
            )
            layout["health"].update(health_panel)
            
            # Footer
            footer_text = Text(
                f"Qdrant Vector Database Monitor | "
                f"Updated: {datetime.now().strftime('%H:%M:%S')}",
                justify="center", 
                style="dim"
            )
            layout["footer"].update(Panel(footer_text, border_style="dim"))
            
            # Print the dashboard
            self.console.print(layout)
            
        except Exception as e:
            logger.error(f"Error creating rich dashboard: {e}")
            self.print_status_simple()
    
    def print_status(self, rich_output: bool = True) -> None:
        """상태 출력"""
        if rich_output and HAS_RICH:
            self.print_status_rich()
        else:
            self.print_status_simple()
    
    def export_status_json(self, output_path: str = None) -> str:
        """상태를 JSON 파일로 출력"""
        try:
            server_stats = self.get_all_collections_stats()
            health_status = self.get_health_status()
            server_info = self.get_server_info()
            
            # Convert dataclasses to dict
            collections_data = []
            for collection in server_stats.collections:
                collections_data.append({
                    'name': collection.name,
                    'vectors_count': collection.vectors_count,
                    'indexed_vectors_count': collection.indexed_vectors_count,
                    'points_count': collection.points_count,
                    'segments_count': collection.segments_count,
                    'status': collection.status,
                    'config': collection.config,
                    'optimizer_status': collection.optimizer_status
                })
            
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'server_info': server_info,
                'server_stats': {
                    'total_collections': server_stats.total_collections,
                    'total_vectors': server_stats.total_vectors,
                    'total_points': server_stats.total_points,
                    'server_version': server_stats.server_version,
                    'last_updated': server_stats.last_updated
                },
                'collections': collections_data,
                'health_status': health_status
            }
            
            if output_path is None:
                output_path = f"qdrant_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
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
                try:
                    server_stats = self.get_all_collections_stats()
                    health = self.get_health_status()
                    
                    # Create simplified live display
                    table = Table(title=f"🐳 Live Qdrant Monitor - {self.host}:{self.port}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="magenta")
                    
                    table.add_row("Collections", f"{server_stats.total_collections}")
                    table.add_row("Total Vectors", f"{server_stats.total_vectors:,}")
                    table.add_row("Total Points", f"{server_stats.total_points:,}")
                    table.add_row("Health", health['status'].upper())
                    table.add_row("Last Update", datetime.now().strftime('%H:%M:%S'))
                    
                    # Add collection details
                    if server_stats.collections:
                        table.add_row("", "")  # Separator
                        for collection in server_stats.collections[:3]:  # Show max 3 collections
                            table.add_row(
                                f"📚 {collection.name}",
                                f"{collection.vectors_count:,} vectors"
                            )
                    
                    yield Panel(table, title="🔄 Live Monitor", border_style="blue")
                
                except Exception as e:
                    error_table = Table(title="❌ Error")
                    error_table.add_column("Error", style="red")
                    error_table.add_row(str(e))
                    yield Panel(error_table, border_style="red")
                
                time.sleep(interval)
        
        try:
            with Live(generate_display(), refresh_per_second=1/interval) as live:
                while True:
                    time.sleep(interval)
        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n👋 Monitoring stopped")

# CLI Interface
def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qdrant Vector Database Status Monitor")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument("--export", help="Export status to JSON file")
    parser.add_argument("--watch", action="store_true", help="Enable live monitoring")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds")
    parser.add_argument("--simple", action="store_true", help="Use simple text output")
    parser.add_argument("--test", action="store_true", help="Test connection only")
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = QdrantStatusMonitor(args.host, args.port, args.api_key)
        
        if args.test:
            # Test connection
            if monitor.test_connection():
                print("✅ Connection successful!")
                server_info = monitor.get_server_info()
                print(f"📦 Server version: {server_info['version']}")
            else:
                print("❌ Connection failed!")
                return 1
                
        elif args.watch:
            print("🔄 Starting live monitoring... (Press Ctrl+C to stop)")
            monitor.watch_status(args.interval)
            
        elif args.export:
            output_file = monitor.export_status_json(args.export)
            print(f"📄 Status exported to {output_file}")
            
        else:
            # Print status
            monitor.print_status(rich_output=not args.simple)
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())