#!/usr/bin/env python3
"""
Real-time PostgreSQL + pgvector Database Monitor

This GUI application provides real-time monitoring for the pgvector database including:
- Active database connections and user information
- Total vector count with real-time updates
- Clickable vector list with detailed metadata viewing
- Original image/file preview for each vector
- Database statistics and performance metrics
- Works while embedding processing is ongoing

Dependencies:
- tkinter for GUI
- psycopg2 for database connectivity
- PIL/Pillow for image display
- python-dotenv for configuration
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import psycopg2
from psycopg2 import pool
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading
import time
from PIL import Image, ImageTk
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """
    Database monitoring backend

    Handles all database queries for monitoring including:
    - Connection information
    - Vector counts
    - Active queries
    - Database statistics
    """

    def __init__(self):
        self.db_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'vector_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        self.connection_pool = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize database connection pool"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 5, **self.db_params
            )

            if self.connection_pool:
                # Test connection
                conn = self.connection_pool.getconn()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    cursor.close()
                    logger.info(f"Connected to: {version[:50]}...")
                    self.initialized = True
                    return True
                finally:
                    self.connection_pool.putconn(conn)

            return False

        except Exception as e:
            logger.error(f"Failed to initialize database monitor: {e}")
            return False

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about active database connections"""
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            query = """
                SELECT
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    query_start,
                    state_change,
                    query
                FROM pg_stat_activity
                WHERE datname = %s
                ORDER BY pid
            """

            cursor.execute(query, (self.db_params['database'],))
            results = cursor.fetchall()

            connections = []
            for row in results:
                connections.append({
                    'pid': row[0],
                    'user': row[1],
                    'application': row[2],
                    'client_addr': str(row[3]) if row[3] else 'local',
                    'state': row[4],
                    'query_start': row[5],
                    'state_change': row[6],
                    'query': row[7][:100] if row[7] else ''  # Truncate long queries
                })

            cursor.close()
            return connections

        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return []

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_vector_count(self) -> int:
        """Get total number of vectors in database"""
        if not self.initialized:
            return 0

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL")
            count = cursor.fetchone()[0]

            cursor.close()
            return count

        except Exception as e:
            logger.error(f"Error getting vector count: {e}")
            return 0

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_total_faces(self) -> int:
        """Get total number of faces (with or without embeddings)"""
        if not self.initialized:
            return 0

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM faces")
            count = cursor.fetchone()[0]

            cursor.close()
            return count

        except Exception as e:
            logger.error(f"Error getting total faces: {e}")
            return 0

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if not self.initialized:
            return {}

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            # Get statistics
            stats = {}

            # Total faces
            cursor.execute("SELECT COUNT(*) FROM faces")
            stats['total_faces'] = cursor.fetchone()[0]

            # Faces with embeddings
            cursor.execute("SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL")
            stats['faces_with_embeddings'] = cursor.fetchone()[0]

            # Embedding models used
            cursor.execute("""
                SELECT embedding_model, COUNT(*)
                FROM faces
                WHERE embedding_model IS NOT NULL
                GROUP BY embedding_model
            """)
            stats['embedding_models'] = dict(cursor.fetchall())

            # Date range
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM faces")
            date_range = cursor.fetchone()
            stats['oldest_face'] = date_range[0]
            stats['newest_face'] = date_range[1]

            # Database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(%s))
            """, (self.db_params['database'],))
            stats['database_size'] = cursor.fetchone()[0]

            # Table size
            cursor.execute("""
                SELECT pg_size_pretty(pg_total_relation_size('faces'))
            """)
            stats['table_size'] = cursor.fetchone()[0]

            cursor.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_face_list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get list of faces with basic info"""
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            query = """
                SELECT
                    face_id,
                    file_path,
                    timestamp,
                    image_hash,
                    embedding_model,
                    created_at,
                    CASE WHEN embedding IS NOT NULL THEN TRUE ELSE FALSE END as has_embedding
                FROM faces
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """

            cursor.execute(query, (limit, offset))
            results = cursor.fetchall()

            faces = []
            for row in results:
                faces.append({
                    'face_id': row[0],
                    'file_path': row[1],
                    'timestamp': row[2],
                    'image_hash': row[3],
                    'embedding_model': row[4],
                    'created_at': row[5],
                    'has_embedding': row[6]
                })

            cursor.close()
            return faces

        except Exception as e:
            logger.error(f"Error getting face list: {e}")
            return []

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def get_face_details(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific face"""
        if not self.initialized:
            return None

        conn = None
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()

            query = """
                SELECT
                    face_id, file_path, timestamp, image_hash, embedding_model,
                    age_estimate, gender, brightness, contrast, sharpness,
                    metadata, created_at, updated_at,
                    embedding
                FROM faces
                WHERE face_id = %s
            """

            cursor.execute(query, (face_id,))
            row = cursor.fetchone()

            if not row:
                cursor.close()
                return None

            # Build comprehensive details
            details = {
                'face_id': row[0],
                'file_path': row[1],
                'timestamp': row[2],
                'image_hash': row[3],
                'embedding_model': row[4],
                'age_estimate': row[5],
                'gender': row[6],
                'brightness': row[7],
                'contrast': row[8],
                'sharpness': row[9],
                'metadata': row[10] if row[10] else {},
                'created_at': row[11],
                'updated_at': row[12],
                'has_embedding': row[13] is not None,
                'embedding_dimension': len(row[13]) if row[13] else 0
            }

            cursor.close()
            return details

        except Exception as e:
            logger.error(f"Error getting face details: {e}")
            return None

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def close(self):
        """Close database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connections closed")


class MonitorGUI:
    """
    Main GUI application for database monitoring

    Provides a comprehensive interface for monitoring the pgvector database
    with real-time updates and detailed information viewing.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("pgvector Database Monitor")
        self.root.geometry("1400x900")

        # Initialize database monitor
        self.db_monitor = DatabaseMonitor()
        if not self.db_monitor.initialize():
            messagebox.showerror("Error", "Failed to connect to database. Check your configuration.")
            self.root.destroy()
            return

        # Monitor state
        self.monitoring = True
        self.auto_refresh = True
        self.refresh_interval = 2000  # milliseconds
        self.current_face_id = None
        self.current_offset = 0
        self.faces_per_page = 50

        # Setup GUI
        self.setup_gui()

        # Start monitoring
        self.start_monitoring()

    def setup_gui(self):
        """Setup the GUI layout"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Overview
        self.overview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_tab, text="Overview")
        self.setup_overview_tab()

        # Tab 2: Connections
        self.connections_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.connections_tab, text="Connections")
        self.setup_connections_tab()

        # Tab 3: Vectors
        self.vectors_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vectors_tab, text="Vectors")
        self.setup_vectors_tab()

        # Tab 4: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.setup_stats_tab()

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Connected", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_overview_tab(self):
        """Setup overview tab with key metrics"""
        # Title
        title = ttk.Label(self.overview_tab, text="Database Overview", font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Metrics frame
        metrics_frame = ttk.LabelFrame(self.overview_tab, text="Key Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Grid for metrics
        self.metric_labels = {}

        metrics = [
            ("Total Faces", "total_faces"),
            ("Vectors (with embeddings)", "vectors"),
            ("Active Connections", "connections"),
            ("Database Size", "db_size"),
            ("Table Size", "table_size"),
            ("Last Updated", "last_update")
        ]

        for idx, (label, key) in enumerate(metrics):
            row = idx // 2
            col = (idx % 2) * 2

            ttk.Label(metrics_frame, text=label + ":", font=('Arial', 11, 'bold')).grid(
                row=row, column=col, sticky=tk.W, padx=10, pady=5
            )

            value_label = ttk.Label(metrics_frame, text="--", font=('Arial', 11))
            value_label.grid(row=row, column=col+1, sticky=tk.W, padx=10, pady=5)
            self.metric_labels[key] = value_label

        # Control frame
        control_frame = ttk.Frame(self.overview_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Auto-refresh",
            variable=self.auto_refresh_var,
            command=self.toggle_auto_refresh
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Refresh Now", command=self.manual_refresh).pack(side=tk.LEFT, padx=5)

        # Refresh interval
        ttk.Label(control_frame, text="Interval (s):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="2")
        interval_entry = ttk.Entry(control_frame, textvariable=self.interval_var, width=5)
        interval_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Set", command=self.set_refresh_interval).pack(side=tk.LEFT, padx=5)

    def setup_connections_tab(self):
        """Setup connections tab"""
        # Title
        title = ttk.Label(self.connections_tab, text="Active Database Connections", font=('Arial', 14, 'bold'))
        title.pack(pady=10)

        # Treeview for connections
        columns = ('PID', 'User', 'Application', 'Client', 'State', 'Query Start')
        self.connections_tree = ttk.Treeview(self.connections_tab, columns=columns, show='headings', height=20)

        for col in columns:
            self.connections_tree.heading(col, text=col)
            self.connections_tree.column(col, width=150)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.connections_tab, orient=tk.VERTICAL, command=self.connections_tree.yview)
        self.connections_tree.configure(yscrollcommand=scrollbar.set)

        self.connections_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=5)

        # Bind double-click to show query details
        self.connections_tree.bind('<Double-1>', self.show_query_details)

    def setup_vectors_tab(self):
        """Setup vectors tab with list and detail view"""
        # Create paned window for split view
        paned = ttk.PanedWindow(self.vectors_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Vector list
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Vector List", font=('Arial', 12, 'bold')).pack(pady=5)

        # Navigation frame
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_page).pack(side=tk.LEFT, padx=2)
        self.page_label = ttk.Label(nav_frame, text="Page 1")
        self.page_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Refresh", command=self.refresh_face_list).pack(side=tk.LEFT, padx=10)

        # Treeview for face list
        columns = ('Face ID', 'Model', 'Created', 'Has Vector')
        self.faces_tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=25)

        self.faces_tree.heading('Face ID', text='Face ID')
        self.faces_tree.column('Face ID', width=200)

        self.faces_tree.heading('Model', text='Model')
        self.faces_tree.column('Model', width=100)

        self.faces_tree.heading('Created', text='Created')
        self.faces_tree.column('Created', width=150)

        self.faces_tree.heading('Has Vector', text='Vector')
        self.faces_tree.column('Has Vector', width=60)

        # Scrollbar
        scrollbar_left = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.faces_tree.yview)
        self.faces_tree.configure(yscrollcommand=scrollbar_left.set)

        self.faces_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection
        self.faces_tree.bind('<<TreeviewSelect>>', self.on_face_select)

        # Right panel: Detail view
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)

        ttk.Label(right_frame, text="Vector Details", font=('Arial', 12, 'bold')).pack(pady=5)

        # Create notebook for details
        detail_notebook = ttk.Notebook(right_frame)
        detail_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Image tab
        image_frame = ttk.Frame(detail_notebook)
        detail_notebook.add(image_frame, text="Image")

        self.image_label = ttk.Label(image_frame, text="Select a face to view image")
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Metadata tab
        metadata_frame = ttk.Frame(detail_notebook)
        detail_notebook.add(metadata_frame, text="Metadata")

        self.metadata_text = scrolledtext.ScrolledText(metadata_frame, wrap=tk.WORD, font=('Courier', 9))
        self.metadata_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Vector info tab
        vector_frame = ttk.Frame(detail_notebook)
        detail_notebook.add(vector_frame, text="Vector Info")

        self.vector_text = scrolledtext.ScrolledText(vector_frame, wrap=tk.WORD, font=('Courier', 9))
        self.vector_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_stats_tab(self):
        """Setup statistics tab"""
        ttk.Label(self.stats_tab, text="Database Statistics", font=('Arial', 14, 'bold')).pack(pady=10)

        # Stats text area
        self.stats_text = scrolledtext.ScrolledText(self.stats_tab, wrap=tk.WORD, font=('Courier', 10))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Refresh button
        ttk.Button(self.stats_tab, text="Refresh Statistics", command=self.refresh_stats).pack(pady=5)

    def start_monitoring(self):
        """Start the monitoring loop"""
        self.update_overview()
        self.update_connections()
        self.refresh_face_list()
        self.refresh_stats()

    def update_overview(self):
        """Update overview metrics"""
        if not self.monitoring:
            return

        try:
            # Get metrics
            total_faces = self.db_monitor.get_total_faces()
            vector_count = self.db_monitor.get_vector_count()
            connections = len(self.db_monitor.get_connection_info())
            stats = self.db_monitor.get_database_stats()

            # Update labels
            self.metric_labels['total_faces'].config(text=str(total_faces))
            self.metric_labels['vectors'].config(text=str(vector_count))
            self.metric_labels['connections'].config(text=str(connections))
            self.metric_labels['db_size'].config(text=stats.get('database_size', '--'))
            self.metric_labels['table_size'].config(text=stats.get('table_size', '--'))
            self.metric_labels['last_update'].config(text=datetime.now().strftime('%H:%M:%S'))

            # Update status bar
            self.status_bar.config(text=f"Connected | Total: {total_faces} | Vectors: {vector_count} | Last update: {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            logger.error(f"Error updating overview: {e}")
            self.status_bar.config(text=f"Error: {e}")

        # Schedule next update
        if self.auto_refresh:
            self.root.after(self.refresh_interval, self.update_overview)

    def update_connections(self):
        """Update connections list"""
        if not self.monitoring:
            return

        try:
            # Clear existing items
            for item in self.connections_tree.get_children():
                self.connections_tree.delete(item)

            # Get connections
            connections = self.db_monitor.get_connection_info()

            # Add to tree
            for conn in connections:
                query_start = conn['query_start'].strftime('%H:%M:%S') if conn['query_start'] else '--'

                self.connections_tree.insert('', tk.END, values=(
                    conn['pid'],
                    conn['user'],
                    conn['application'],
                    conn['client_addr'],
                    conn['state'],
                    query_start
                ))

        except Exception as e:
            logger.error(f"Error updating connections: {e}")

        # Schedule next update
        if self.auto_refresh:
            self.root.after(self.refresh_interval, self.update_connections)

    def refresh_face_list(self):
        """Refresh the face list"""
        try:
            # Clear existing items
            for item in self.faces_tree.get_children():
                self.faces_tree.delete(item)

            # Get faces
            faces = self.db_monitor.get_face_list(self.faces_per_page, self.current_offset)

            # Add to tree
            for face in faces:
                created = face['created_at'].strftime('%Y-%m-%d %H:%M') if face['created_at'] else '--'
                has_vector = '✓' if face['has_embedding'] else '✗'

                self.faces_tree.insert('', tk.END, values=(
                    face['face_id'],
                    face['embedding_model'] or '--',
                    created,
                    has_vector
                ))

            # Update page label
            page_num = (self.current_offset // self.faces_per_page) + 1
            self.page_label.config(text=f"Page {page_num}")

        except Exception as e:
            logger.error(f"Error refreshing face list: {e}")

    def previous_page(self):
        """Go to previous page"""
        if self.current_offset >= self.faces_per_page:
            self.current_offset -= self.faces_per_page
            self.refresh_face_list()

    def next_page(self):
        """Go to next page"""
        self.current_offset += self.faces_per_page
        self.refresh_face_list()

    def on_face_select(self, event):
        """Handle face selection"""
        selection = self.faces_tree.selection()
        if not selection:
            return

        item = self.faces_tree.item(selection[0])
        face_id = item['values'][0]

        self.show_face_details(face_id)

    def show_face_details(self, face_id: str):
        """Show detailed information for a face"""
        try:
            details = self.db_monitor.get_face_details(face_id)

            if not details:
                messagebox.showwarning("Not Found", f"Face {face_id} not found")
                return

            self.current_face_id = face_id

            # Display image
            self.display_image(details['file_path'])

            # Display metadata
            self.metadata_text.delete('1.0', tk.END)
            metadata_str = json.dumps(details['metadata'], indent=2, default=str)
            self.metadata_text.insert('1.0', f"Face ID: {details['face_id']}\n")
            self.metadata_text.insert(tk.END, f"File Path: {details['file_path']}\n")
            self.metadata_text.insert(tk.END, f"Image Hash: {details['image_hash']}\n")
            self.metadata_text.insert(tk.END, f"Timestamp: {details['timestamp']}\n")
            self.metadata_text.insert(tk.END, f"Created: {details['created_at']}\n")
            self.metadata_text.insert(tk.END, f"Updated: {details['updated_at']}\n")
            self.metadata_text.insert(tk.END, f"\nStructured Metadata:\n")
            self.metadata_text.insert(tk.END, f"Age Estimate: {details['age_estimate']}\n")
            self.metadata_text.insert(tk.END, f"Gender: {details['gender']}\n")
            self.metadata_text.insert(tk.END, f"Brightness: {details['brightness']}\n")
            self.metadata_text.insert(tk.END, f"Contrast: {details['contrast']}\n")
            self.metadata_text.insert(tk.END, f"Sharpness: {details['sharpness']}\n")
            self.metadata_text.insert(tk.END, f"\nFull Metadata (JSONB):\n{metadata_str}")

            # Display vector info
            self.vector_text.delete('1.0', tk.END)
            self.vector_text.insert('1.0', f"Embedding Model: {details['embedding_model']}\n")
            self.vector_text.insert(tk.END, f"Has Embedding: {details['has_embedding']}\n")
            self.vector_text.insert(tk.END, f"Embedding Dimension: {details['embedding_dimension']}\n")

        except Exception as e:
            logger.error(f"Error showing face details: {e}")
            messagebox.showerror("Error", f"Failed to load face details: {e}")

    def display_image(self, file_path: str):
        """Display image in the image label"""
        try:
            if not os.path.exists(file_path):
                self.image_label.config(text=f"Image not found:\n{file_path}", image='')
                return

            # Load and resize image
            image = Image.open(file_path)

            # Resize to fit in display area (max 400x400)
            max_size = (400, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update label
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo  # Keep a reference

        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            self.image_label.config(text=f"Error loading image:\n{e}", image='')

    def refresh_stats(self):
        """Refresh statistics"""
        try:
            stats = self.db_monitor.get_database_stats()

            self.stats_text.delete('1.0', tk.END)

            self.stats_text.insert('1.0', "=" * 60 + "\n")
            self.stats_text.insert(tk.END, "DATABASE STATISTICS\n")
            self.stats_text.insert(tk.END, "=" * 60 + "\n\n")

            self.stats_text.insert(tk.END, f"Total Faces: {stats.get('total_faces', 0)}\n")
            self.stats_text.insert(tk.END, f"Faces with Embeddings: {stats.get('faces_with_embeddings', 0)}\n")
            self.stats_text.insert(tk.END, f"Database Size: {stats.get('database_size', '--')}\n")
            self.stats_text.insert(tk.END, f"Table Size: {stats.get('table_size', '--')}\n\n")

            self.stats_text.insert(tk.END, "Date Range:\n")
            oldest = stats.get('oldest_face')
            newest = stats.get('newest_face')
            self.stats_text.insert(tk.END, f"  Oldest: {oldest.strftime('%Y-%m-%d %H:%M:%S') if oldest else '--'}\n")
            self.stats_text.insert(tk.END, f"  Newest: {newest.strftime('%Y-%m-%d %H:%M:%S') if newest else '--'}\n\n")

            self.stats_text.insert(tk.END, "Embedding Models:\n")
            for model, count in stats.get('embedding_models', {}).items():
                self.stats_text.insert(tk.END, f"  {model}: {count}\n")

            self.stats_text.insert(tk.END, f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        except Exception as e:
            logger.error(f"Error refreshing stats: {e}")

    def show_query_details(self, event):
        """Show full query details when connection is double-clicked"""
        selection = self.connections_tree.selection()
        if not selection:
            return

        item = self.connections_tree.item(selection[0])
        pid = item['values'][0]

        # Get full connection info
        connections = self.db_monitor.get_connection_info()
        conn_info = next((c for c in connections if c['pid'] == pid), None)

        if conn_info:
            details = f"PID: {conn_info['pid']}\n"
            details += f"User: {conn_info['user']}\n"
            details += f"Application: {conn_info['application']}\n"
            details += f"Client: {conn_info['client_addr']}\n"
            details += f"State: {conn_info['state']}\n"
            details += f"Query Start: {conn_info['query_start']}\n"
            details += f"\nQuery:\n{conn_info['query']}"

            # Create popup window
            popup = tk.Toplevel(self.root)
            popup.title(f"Connection Details - PID {pid}")
            popup.geometry("600x400")

            text = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=('Courier', 9))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert('1.0', details)
            text.config(state=tk.DISABLED)

    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.start_monitoring()

    def manual_refresh(self):
        """Manual refresh trigger"""
        self.update_overview()
        self.update_connections()
        self.refresh_face_list()
        self.refresh_stats()

    def set_refresh_interval(self):
        """Set new refresh interval"""
        try:
            interval_sec = float(self.interval_var.get())
            if interval_sec < 0.5:
                interval_sec = 0.5
            if interval_sec > 60:
                interval_sec = 60

            self.refresh_interval = int(interval_sec * 1000)
            messagebox.showinfo("Success", f"Refresh interval set to {interval_sec} seconds")
        except ValueError:
            messagebox.showerror("Error", "Invalid interval value")

    def on_closing(self):
        """Handle window closing"""
        self.monitoring = False
        self.db_monitor.close()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MonitorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
