#!/usr/bin/env python3
"""
Enhanced PostgreSQL + pgvector Database Monitor with Query Interface

Features:
- Cleaner, information-focused UI
- Vector similarity search
- Metadata filtering and queries
- SQL console for advanced users
- Real-time monitoring
- Image viewing with metadata
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import psycopg2
from psycopg2 import pool
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import threading
import time
from PIL import Image, ImageTk
from dotenv import load_dotenv
import logging
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseMonitor:
    """Enhanced Database monitoring backend with query capabilities"""

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
                1, 10, **self.db_params
            )

            if self.connection_pool:
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

    def get_connection(self):
        """Get connection from pool"""
        if not self.connection_pool:
            raise Exception("Connection pool not initialized")
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    def get_vector_count(self) -> int:
        """Get total number of vectors"""
        if not self.initialized:
            return 0

        conn = None
        try:
            conn = self.get_connection()
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
                self.return_connection(conn)

    def get_total_faces(self) -> int:
        """Get total number of faces"""
        if not self.initialized:
            return 0

        conn = None
        try:
            conn = self.get_connection()
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
                self.return_connection(conn)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if not self.initialized:
            return {}

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            stats = {}

            # Total faces
            cursor.execute("SELECT COUNT(*) FROM faces")
            stats['total_faces'] = cursor.fetchone()[0]

            # Faces with embeddings
            cursor.execute("SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL")
            stats['faces_with_embeddings'] = cursor.fetchone()[0]

            # Embedding models
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
            cursor.execute("SELECT pg_size_pretty(pg_database_size(%s))", (self.db_params['database'],))
            stats['database_size'] = cursor.fetchone()[0]

            # Table size
            cursor.execute("SELECT pg_size_pretty(pg_total_relation_size('faces'))")
            stats['table_size'] = cursor.fetchone()[0]

            cursor.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            if conn:
                self.return_connection(conn)

    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search faces by metadata filters

        filters can include:
        - gender: 'male', 'female'
        - age_min, age_max: numeric ranges
        - brightness_min, brightness_max: numeric ranges
        - embedding_model: model name
        """
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
                SELECT face_id, file_path, timestamp, image_hash, embedding_model,
                       age_estimate, gender, brightness, contrast, sharpness,
                       metadata, created_at
                FROM faces
                WHERE 1=1
            """

            params = []

            # Gender filter
            if filters.get('gender'):
                query += " AND (gender = %s OR metadata->>'estimated_sex' = %s)"
                params.extend([filters['gender'], filters['gender']])

            # Age range
            if filters.get('age_min') is not None:
                query += " AND (age_estimate >= %s OR (metadata->>'age_estimate')::float >= %s)"
                params.extend([filters['age_min'], filters['age_min']])

            if filters.get('age_max') is not None:
                query += " AND (age_estimate <= %s OR (metadata->>'age_estimate')::float <= %s)"
                params.extend([filters['age_max'], filters['age_max']])

            # Brightness range
            if filters.get('brightness_min') is not None:
                query += " AND brightness >= %s"
                params.append(filters['brightness_min'])

            if filters.get('brightness_max') is not None:
                query += " AND brightness <= %s"
                params.append(filters['brightness_max'])

            # Embedding model
            if filters.get('embedding_model'):
                query += " AND embedding_model = %s"
                params.append(filters['embedding_model'])

            # Skin tone
            if filters.get('skin_tone'):
                query += " AND metadata->>'skin_tone' = %s"
                params.append(filters['skin_tone'])

            # Hair color
            if filters.get('hair_color'):
                query += " AND metadata->>'hair_color' = %s"
                params.append(filters['hair_color'])

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            faces = []
            for row in results:
                faces.append({
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
                    'metadata': row[10] or {},
                    'created_at': row[11]
                })

            cursor.close()
            return faces

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)

    def get_random_face(self) -> Optional[Dict[str, Any]]:
        """Get a random face for similarity search"""
        if not self.initialized:
            return None

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT face_id, file_path, embedding
                FROM faces
                WHERE embedding IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                return {
                    'face_id': row[0],
                    'file_path': row[1],
                    'embedding': row[2]
                }

            cursor.close()
            return None

        except Exception as e:
            logger.error(f"Error getting random face: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def get_face_embedding(self, face_id: str) -> Optional[List[float]]:
        """Get embedding for a specific face"""
        if not self.initialized:
            return None

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT embedding FROM faces WHERE face_id = %s", (face_id,))
            row = cursor.fetchone()

            cursor.close()
            return row[0] if row else None

        except Exception as e:
            logger.error(f"Error getting face embedding: {e}")
            return None
        finally:
            if conn:
                self.return_connection(conn)

    def similarity_search(self, embedding: List[float], limit: int = 10,
                         distance_metric: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Perform vector similarity search

        distance_metric: 'cosine', 'l2', or 'inner_product'
        """
        if not self.initialized:
            return []

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Select distance operator
            if distance_metric == 'cosine':
                distance_op = '<=>'
            elif distance_metric == 'l2':
                distance_op = '<->'
            elif distance_metric == 'inner_product':
                distance_op = '<#>'
            else:
                distance_op = '<=>'

            query = f"""
                SELECT face_id, file_path, timestamp, image_hash, embedding_model,
                       age_estimate, gender, brightness, metadata,
                       embedding {distance_op} %s::vector AS distance
                FROM faces
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
            """

            cursor.execute(query, (embedding, limit))
            results = cursor.fetchall()

            faces = []
            for row in results:
                faces.append({
                    'face_id': row[0],
                    'file_path': row[1],
                    'timestamp': row[2],
                    'image_hash': row[3],
                    'embedding_model': row[4],
                    'age_estimate': row[5],
                    'gender': row[6],
                    'brightness': row[7],
                    'metadata': row[8] or {},
                    'distance': float(row[9])
                })

            cursor.close()
            return faces

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)

    def execute_custom_query(self, query: str) -> Tuple[List[tuple], List[str], Optional[str]]:
        """
        Execute custom SQL query
        Returns: (rows, column_names, error_message)
        """
        if not self.initialized:
            return [], [], "Database not initialized"

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(query)

            # Check if query returns results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
            else:
                columns = []
                rows = []

            conn.commit()
            cursor.close()

            return rows, columns, None

        except Exception as e:
            error_msg = f"Query error: {e}"
            logger.error(error_msg)
            if conn:
                conn.rollback()
            return [], [], error_msg
        finally:
            if conn:
                self.return_connection(conn)

    def get_face_details(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific face"""
        if not self.initialized:
            return None

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = """
                SELECT face_id, file_path, timestamp, image_hash, embedding_model,
                       age_estimate, gender, brightness, contrast, sharpness,
                       metadata, created_at, updated_at, embedding
                FROM faces
                WHERE face_id = %s
            """

            cursor.execute(query, (face_id,))
            row = cursor.fetchone()

            if not row:
                cursor.close()
                return None

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
                self.return_connection(conn)

    def close(self):
        """Close database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connections closed")


class EnhancedMonitorGUI:
    """Enhanced GUI with cleaner design and query interface"""

    def __init__(self, root):
        self.root = root
        self.root.title("pgvector Monitor & Query Tool")
        self.root.geometry("1600x950")

        # Initialize database monitor
        self.db_monitor = DatabaseMonitor()
        if not self.db_monitor.initialize():
            messagebox.showerror("Error", "Failed to connect to database. Check your configuration.")
            self.root.destroy()
            return

        # State
        self.monitoring = True
        self.auto_refresh = True
        self.refresh_interval = 3000  # 3 seconds
        self.current_results = []
        self.current_image_cache = {}

        # Setup GUI
        self.setup_gui()

        # Start monitoring
        self.start_monitoring()

    def setup_gui(self):
        """Setup the GUI with cleaner design"""

        # Top status bar - CLEAN AND SIMPLE
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        # Simple status labels
        self.status_labels = {}

        status_items = [
            ('Total Faces', 'total'),
            ('Vectors', 'vectors'),
            ('DB Size', 'db_size'),
            ('Last Update', 'last_update')
        ]

        for idx, (label, key) in enumerate(status_items):
            frame = ttk.Frame(status_frame)
            frame.pack(side=tk.LEFT, padx=15)

            ttk.Label(frame, text=label + ":", font=('Arial', 9)).pack(side=tk.LEFT)
            value_label = ttk.Label(frame, text="--", font=('Arial', 9, 'bold'), foreground='blue')
            value_label.pack(side=tk.LEFT, padx=5)
            self.status_labels[key] = value_label

        # Auto-refresh toggle
        refresh_frame = ttk.Frame(status_frame)
        refresh_frame.pack(side=tk.RIGHT, padx=10)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(refresh_frame, text="Auto-refresh", variable=self.auto_refresh_var,
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT)

        ttk.Button(refresh_frame, text="↻ Refresh", command=self.manual_refresh).pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=2)

        # Main notebook with 3 tabs - SIMPLER
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Query Interface
        self.query_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.query_tab, text="🔍 Query & Search")
        self.setup_query_tab()

        # Tab 2: Statistics (simpler)
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="📊 Statistics")
        self.setup_stats_tab()

        # Tab 3: SQL Console
        self.sql_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sql_tab, text="⚡ SQL Console")
        self.setup_sql_tab()

    def setup_query_tab(self):
        """Setup query and search interface"""

        # Create horizontal paned window
        paned = ttk.PanedWindow(self.query_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # LEFT PANEL: Query controls
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        # Query type selector
        ttk.Label(left_panel, text="Query Type", font=('Arial', 11, 'bold')).pack(pady=5)

        self.query_type = tk.StringVar(value='metadata')

        query_types = [
            ('Metadata Search', 'metadata'),
            ('Vector Similarity', 'similarity'),
        ]

        for text, value in query_types:
            ttk.Radiobutton(left_panel, text=text, variable=self.query_type,
                           value=value, command=self.on_query_type_change).pack(anchor=tk.W, padx=20)

        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Metadata filters frame
        self.metadata_frame = ttk.LabelFrame(left_panel, text="Metadata Filters", padding=10)
        self.metadata_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Gender filter
        ttk.Label(self.metadata_frame, text="Gender:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar(value='')
        gender_combo = ttk.Combobox(self.metadata_frame, textvariable=self.gender_var,
                                    values=['', 'male', 'female', 'unknown'], width=15)
        gender_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        # Age range
        ttk.Label(self.metadata_frame, text="Age Range:").grid(row=1, column=0, sticky=tk.W, pady=5)
        age_frame = ttk.Frame(self.metadata_frame)
        age_frame.grid(row=1, column=1, sticky=tk.W, pady=5)

        self.age_min_var = tk.StringVar()
        self.age_max_var = tk.StringVar()
        ttk.Entry(age_frame, textvariable=self.age_min_var, width=5).pack(side=tk.LEFT)
        ttk.Label(age_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(age_frame, textvariable=self.age_max_var, width=5).pack(side=tk.LEFT)

        # Brightness range
        ttk.Label(self.metadata_frame, text="Brightness:").grid(row=2, column=0, sticky=tk.W, pady=5)
        brightness_frame = ttk.Frame(self.metadata_frame)
        brightness_frame.grid(row=2, column=1, sticky=tk.W, pady=5)

        self.brightness_min_var = tk.StringVar()
        self.brightness_max_var = tk.StringVar()
        ttk.Entry(brightness_frame, textvariable=self.brightness_min_var, width=5).pack(side=tk.LEFT)
        ttk.Label(brightness_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(brightness_frame, textvariable=self.brightness_max_var, width=5).pack(side=tk.LEFT)
        ttk.Label(brightness_frame, text=" (0-255)").pack(side=tk.LEFT, padx=5)

        # Skin tone
        ttk.Label(self.metadata_frame, text="Skin Tone:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.skin_tone_var = tk.StringVar(value='')
        skin_combo = ttk.Combobox(self.metadata_frame, textvariable=self.skin_tone_var,
                                  values=['', 'very_light', 'light', 'medium', 'tan', 'brown', 'dark'], width=15)
        skin_combo.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # Hair color
        ttk.Label(self.metadata_frame, text="Hair Color:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.hair_color_var = tk.StringVar(value='')
        hair_combo = ttk.Combobox(self.metadata_frame, textvariable=self.hair_color_var,
                                  values=['', 'black', 'brown', 'dark_brown', 'blonde', 'red', 'gray', 'other'], width=15)
        hair_combo.grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

        # Result limit
        ttk.Label(self.metadata_frame, text="Max Results:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.limit_var = tk.StringVar(value='20')
        ttk.Entry(self.metadata_frame, textvariable=self.limit_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)

        # Similarity search frame
        self.similarity_frame = ttk.LabelFrame(left_panel, text="Similarity Search", padding=10)
        # Will be shown when similarity is selected

        ttk.Label(self.similarity_frame, text="Find faces similar to:").pack(anchor=tk.W, pady=5)

        ttk.Button(self.similarity_frame, text="🎲 Pick Random Face",
                  command=self.pick_random_face).pack(fill=tk.X, pady=5)

        ttk.Label(self.similarity_frame, text="or").pack(pady=5)

        ttk.Button(self.similarity_frame, text="📁 Select from Results Below",
                  command=self.select_from_results).pack(fill=tk.X, pady=5)

        self.selected_face_label = ttk.Label(self.similarity_frame, text="No face selected",
                                             foreground='gray')
        self.selected_face_label.pack(pady=5)

        ttk.Label(self.similarity_frame, text="Distance Metric:").pack(anchor=tk.W, pady=(10, 5))
        self.distance_metric_var = tk.StringVar(value='cosine')
        for metric in ['cosine', 'l2', 'inner_product']:
            ttk.Radiobutton(self.similarity_frame, text=metric, variable=self.distance_metric_var,
                           value=metric).pack(anchor=tk.W, padx=20)

        ttk.Label(self.similarity_frame, text="Top K Results:").pack(anchor=tk.W, pady=5)
        self.similarity_limit_var = tk.StringVar(value='10')
        ttk.Entry(self.similarity_frame, textvariable=self.similarity_limit_var, width=10).pack(anchor=tk.W, pady=5)

        # Search button
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        search_btn = ttk.Button(left_panel, text="🔍 SEARCH", command=self.execute_search)
        search_btn.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(left_panel, text="Clear Filters", command=self.clear_filters).pack(fill=tk.X, padx=20)

        # RIGHT PANEL: Results
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)

        ttk.Label(right_panel, text="Search Results", font=('Arial', 11, 'bold')).pack(pady=5)

        self.results_info_label = ttk.Label(right_panel, text="Enter query parameters and click SEARCH",
                                            foreground='gray')
        self.results_info_label.pack(pady=5)

        # Results in grid layout
        results_canvas_frame = ttk.Frame(right_panel)
        results_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable canvas for results
        self.results_canvas = tk.Canvas(results_canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_canvas_frame, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.results_frame = ttk.Frame(self.results_canvas)

        self.results_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )

        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Export button
        ttk.Button(right_panel, text="💾 Export Results", command=self.export_results).pack(pady=5)

    def setup_stats_tab(self):
        """Setup simplified statistics tab"""

        stats_frame = ttk.Frame(self.stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(stats_frame, text="Database Statistics", font=('Arial', 14, 'bold')).pack(pady=10)

        # Stats display
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD,
                                                     font=('Courier', 10), height=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Button(stats_frame, text="↻ Refresh Statistics",
                  command=self.refresh_stats).pack(pady=5)

    def setup_sql_tab(self):
        """Setup SQL console tab"""

        # Instructions
        ttk.Label(self.sql_tab, text="SQL Console - Execute Custom Queries",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        # Example queries
        examples_frame = ttk.LabelFrame(self.sql_tab, text="Example Queries", padding=10)
        examples_frame.pack(fill=tk.X, padx=10, pady=5)

        examples = [
            ("Vector Count", "SELECT COUNT(*) FROM faces WHERE embedding IS NOT NULL;"),
            ("Recent 10 Faces", "SELECT face_id, gender, brightness, created_at FROM faces ORDER BY created_at DESC LIMIT 10;"),
            ("Gender Distribution", "SELECT gender, COUNT(*) FROM faces GROUP BY gender;"),
            ("Avg Brightness by Gender", "SELECT gender, AVG(brightness) FROM faces WHERE gender IS NOT NULL GROUP BY gender;"),
        ]

        for i, (name, query) in enumerate(examples):
            col = i % 2
            row = i // 2
            btn = ttk.Button(examples_frame, text=name,
                           command=lambda q=query: self.load_example_query(q))
            btn.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)

        # Query input
        ttk.Label(self.sql_tab, text="Query:").pack(anchor=tk.W, padx=10, pady=(10, 5))

        self.sql_query_text = scrolledtext.ScrolledText(self.sql_tab, wrap=tk.WORD,
                                                        font=('Courier', 10), height=8)
        self.sql_query_text.pack(fill=tk.X, padx=10, pady=5)

        # Execute button
        exec_frame = ttk.Frame(self.sql_tab)
        exec_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(exec_frame, text="▶ Execute Query",
                  command=self.execute_sql).pack(side=tk.LEFT, padx=5)
        ttk.Button(exec_frame, text="Clear",
                  command=lambda: self.sql_query_text.delete('1.0', tk.END)).pack(side=tk.LEFT)

        # Results
        ttk.Label(self.sql_tab, text="Results:").pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Results treeview
        results_frame = ttk.Frame(self.sql_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.sql_results_tree = ttk.Treeview(results_frame, show='headings')
        sql_scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL,
                                        command=self.sql_results_tree.yview)
        sql_scrollbar_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL,
                                        command=self.sql_results_tree.xview)

        self.sql_results_tree.configure(yscrollcommand=sql_scrollbar_y.set,
                                       xscrollcommand=sql_scrollbar_x.set)

        self.sql_results_tree.grid(row=0, column=0, sticky='nsew')
        sql_scrollbar_y.grid(row=0, column=1, sticky='ns')
        sql_scrollbar_x.grid(row=1, column=0, sticky='ew')

        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

    def on_query_type_change(self):
        """Handle query type change"""
        if self.query_type.get() == 'metadata':
            self.similarity_frame.pack_forget()
            self.metadata_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            self.metadata_frame.pack_forget()
            self.similarity_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def execute_search(self):
        """Execute the search based on query type"""
        query_type = self.query_type.get()

        if query_type == 'metadata':
            self.search_by_metadata()
        elif query_type == 'similarity':
            self.search_by_similarity()

    def search_by_metadata(self):
        """Execute metadata search"""
        try:
            # Build filters
            filters = {}

            if self.gender_var.get():
                filters['gender'] = self.gender_var.get()

            if self.age_min_var.get():
                filters['age_min'] = float(self.age_min_var.get())

            if self.age_max_var.get():
                filters['age_max'] = float(self.age_max_var.get())

            if self.brightness_min_var.get():
                filters['brightness_min'] = float(self.brightness_min_var.get())

            if self.brightness_max_var.get():
                filters['brightness_max'] = float(self.brightness_max_var.get())

            if self.skin_tone_var.get():
                filters['skin_tone'] = self.skin_tone_var.get()

            if self.hair_color_var.get():
                filters['hair_color'] = self.hair_color_var.get()

            limit = int(self.limit_var.get()) if self.limit_var.get() else 20

            # Execute search
            results = self.db_monitor.search_by_metadata(filters, limit)

            # Display results
            self.display_results(results, f"Metadata Search - {len(results)} results")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input values: {e}")
        except Exception as e:
            messagebox.showerror("Search Error", f"Search failed: {e}")

    def search_by_similarity(self):
        """Execute similarity search"""
        if not hasattr(self, 'selected_embedding') or self.selected_embedding is None:
            messagebox.showwarning("No Face Selected",
                                 "Please select a face for similarity search using the buttons above.")
            return

        try:
            limit = int(self.similarity_limit_var.get()) if self.similarity_limit_var.get() else 10
            metric = self.distance_metric_var.get()

            # Execute similarity search
            results = self.db_monitor.similarity_search(self.selected_embedding, limit, metric)

            # Display results
            self.display_results(results,
                               f"Similarity Search ({metric}) - {len(results)} results")

        except Exception as e:
            messagebox.showerror("Search Error", f"Similarity search failed: {e}")

    def pick_random_face(self):
        """Pick a random face for similarity search"""
        random_face = self.db_monitor.get_random_face()

        if random_face:
            self.selected_embedding = random_face['embedding']
            self.selected_face_id = random_face['face_id']
            self.selected_face_label.config(
                text=f"Selected: {random_face['face_id'][:30]}...",
                foreground='green'
            )
        else:
            messagebox.showwarning("No Faces", "No faces with embeddings found in database")

    def select_from_results(self):
        """Select a face from current results for similarity search"""
        if not self.current_results:
            messagebox.showinfo("No Results", "Please run a metadata search first to select from results")
            return

        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Face for Similarity Search")
        dialog.geometry("400x500")

        ttk.Label(dialog, text="Click a face to select:", font=('Arial', 11, 'bold')).pack(pady=10)

        # List of faces
        listbox = tk.Listbox(dialog, font=('Courier', 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for face in self.current_results:
            listbox.insert(tk.END, f"{face['face_id']} - {face.get('gender', 'N/A')}")

        def on_select():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                selected_face = self.current_results[idx]

                # Get embedding
                embedding = self.db_monitor.get_face_embedding(selected_face['face_id'])
                if embedding:
                    self.selected_embedding = embedding
                    self.selected_face_id = selected_face['face_id']
                    self.selected_face_label.config(
                        text=f"Selected: {selected_face['face_id'][:30]}...",
                        foreground='green'
                    )
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Failed to get embedding for selected face")

        ttk.Button(dialog, text="Select", command=on_select).pack(pady=10)

    def display_results(self, results: List[Dict[str, Any]], title: str):
        """Display search results in grid"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.current_results = results
        self.current_image_cache.clear()

        # Update info label
        self.results_info_label.config(text=title, foreground='black')

        if not results:
            ttk.Label(self.results_frame, text="No results found",
                     font=('Arial', 12), foreground='gray').pack(pady=20)
            return

        # Display in grid (3 columns)
        cols = 3
        for idx, face in enumerate(results):
            row = idx // cols
            col = idx % cols

            # Create frame for each result
            face_frame = ttk.LabelFrame(self.results_frame, text=f"Result {idx + 1}", padding=5)
            face_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')

            # Load and display image
            try:
                if os.path.exists(face['file_path']):
                    img = Image.open(face['file_path'])
                    img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    img_label = ttk.Label(face_frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.pack()

                    self.current_image_cache[idx] = photo
                else:
                    ttk.Label(face_frame, text="Image not found", foreground='red').pack()
            except Exception as e:
                ttk.Label(face_frame, text=f"Error: {e}", foreground='red').pack()

            # Info
            info_text = f"ID: {face['face_id'][:20]}...\n"
            info_text += f"Gender: {face.get('gender', 'N/A')}\n"
            info_text += f"Age: {face.get('age_estimate', 'N/A')}\n"
            info_text += f"Brightness: {face.get('brightness', 'N/A'):.1f}\n" if face.get('brightness') else "Brightness: N/A\n"

            if 'distance' in face:
                info_text += f"Distance: {face['distance']:.4f}"

            ttk.Label(face_frame, text=info_text, font=('Courier', 8)).pack()

            # View details button
            ttk.Button(face_frame, text="View Details",
                      command=lambda f=face: self.show_face_details(f)).pack(pady=2)

        # Configure grid weights
        for i in range(cols):
            self.results_frame.grid_columnconfigure(i, weight=1)

    def show_face_details(self, face: Dict[str, Any]):
        """Show detailed face information in popup"""
        details = self.db_monitor.get_face_details(face['face_id'])

        if not details:
            messagebox.showerror("Error", "Failed to load face details")
            return

        # Create popup
        popup = tk.Toplevel(self.root)
        popup.title(f"Face Details - {face['face_id'][:30]}...")
        popup.geometry("600x700")

        # Image
        try:
            if os.path.exists(details['file_path']):
                img = Image.open(details['file_path'])
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                img_label = ttk.Label(popup, image=photo)
                img_label.image = photo
                img_label.pack(pady=10)
        except:
            pass

        # Details
        details_text = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=('Courier', 9))
        details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text = f"Face ID: {details['face_id']}\n"
        text += f"File: {details['file_path']}\n"
        text += f"Hash: {details['image_hash']}\n"
        text += f"Model: {details['embedding_model']}\n"
        text += f"Created: {details['created_at']}\n\n"
        text += f"--- Attributes ---\n"
        text += f"Gender: {details.get('gender', 'N/A')}\n"
        text += f"Age: {details.get('age_estimate', 'N/A')}\n"
        text += f"Brightness: {details.get('brightness', 'N/A')}\n"
        text += f"Contrast: {details.get('contrast', 'N/A')}\n"
        text += f"Sharpness: {details.get('sharpness', 'N/A')}\n\n"
        text += f"--- Metadata ---\n"
        text += json.dumps(details.get('metadata', {}), indent=2)

        details_text.insert('1.0', text)
        details_text.config(state=tk.DISABLED)

    def clear_filters(self):
        """Clear all filter fields"""
        self.gender_var.set('')
        self.age_min_var.set('')
        self.age_max_var.set('')
        self.brightness_min_var.set('')
        self.brightness_max_var.set('')
        self.skin_tone_var.set('')
        self.hair_color_var.set('')
        self.limit_var.set('20')

    def load_example_query(self, query: str):
        """Load example query into SQL text box"""
        self.sql_query_text.delete('1.0', tk.END)
        self.sql_query_text.insert('1.0', query)

    def execute_sql(self):
        """Execute custom SQL query"""
        query = self.sql_query_text.get('1.0', tk.END).strip()

        if not query:
            messagebox.showwarning("Empty Query", "Please enter a SQL query")
            return

        # Execute query
        rows, columns, error = self.db_monitor.execute_custom_query(query)

        if error:
            messagebox.showerror("Query Error", error)
            return

        # Clear previous results
        self.sql_results_tree.delete(*self.sql_results_tree.get_children())

        # Configure columns
        self.sql_results_tree['columns'] = columns
        for col in columns:
            self.sql_results_tree.heading(col, text=col)
            self.sql_results_tree.column(col, width=150)

        # Add rows
        for row in rows:
            # Convert to strings, handle None
            display_row = [str(val) if val is not None else 'NULL' for val in row]
            self.sql_results_tree.insert('', tk.END, values=display_row)

        messagebox.showinfo("Success", f"Query executed successfully. {len(rows)} rows returned.")

    def export_results(self):
        """Export current results to JSON"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_results, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")

    def refresh_stats(self):
        """Refresh statistics display"""
        try:
            stats = self.db_monitor.get_database_stats()

            self.stats_text.delete('1.0', tk.END)

            text = "=" * 60 + "\n"
            text += "DATABASE STATISTICS\n"
            text += "=" * 60 + "\n\n"

            text += f"Total Faces: {stats.get('total_faces', 0)}\n"
            text += f"Faces with Embeddings: {stats.get('faces_with_embeddings', 0)}\n"
            text += f"Database Size: {stats.get('database_size', 'N/A')}\n"
            text += f"Table Size: {stats.get('table_size', 'N/A')}\n\n"

            text += "Date Range:\n"
            oldest = stats.get('oldest_face')
            newest = stats.get('newest_face')
            text += f"  Oldest: {oldest.strftime('%Y-%m-%d %H:%M:%S') if oldest else 'N/A'}\n"
            text += f"  Newest: {newest.strftime('%Y-%m-%d %H:%M:%S') if newest else 'N/A'}\n\n"

            text += "Embedding Models:\n"
            for model, count in stats.get('embedding_models', {}).items():
                text += f"  {model}: {count}\n"

            text += f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

            self.stats_text.insert('1.0', text)

        except Exception as e:
            logger.error(f"Error refreshing stats: {e}")

    def start_monitoring(self):
        """Start background monitoring"""
        self.update_status_bar()
        self.refresh_stats()

    def update_status_bar(self):
        """Update top status bar"""
        if not self.monitoring:
            return

        try:
            total = self.db_monitor.get_total_faces()
            vectors = self.db_monitor.get_vector_count()
            stats = self.db_monitor.get_database_stats()

            self.status_labels['total'].config(text=str(total))
            self.status_labels['vectors'].config(text=str(vectors))
            self.status_labels['db_size'].config(text=stats.get('database_size', '--'))
            self.status_labels['last_update'].config(text=datetime.now().strftime('%H:%M:%S'))

        except Exception as e:
            logger.error(f"Error updating status: {e}")

        # Schedule next update
        if self.auto_refresh:
            self.root.after(self.refresh_interval, self.update_status_bar)

    def toggle_auto_refresh(self):
        """Toggle auto-refresh"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.update_status_bar()

    def manual_refresh(self):
        """Manual refresh"""
        self.update_status_bar()
        self.refresh_stats()

    def on_closing(self):
        """Handle window closing"""
        self.monitoring = False
        self.db_monitor.close()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = EnhancedMonitorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
