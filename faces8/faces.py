#!/usr/bin/env python3
"""
Face Processing GUI Application

A comprehensive graphical interface for face image processing including:
- Face image downloading from AI generation services
- Face detection and analysis
- Vector embedding generation
- Similarity search and metadata filtering

This application provides a complete workflow for building and querying
a face recognition database using PostgreSQL with pgvector and various embedding models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import json
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import hashlib
from pathlib import Path
from PIL import Image, ImageTk
import io
import numpy as np
import numpy as np

# Import core backend functionality
try:
    from core import (
        IntegratedFaceSystem, SystemConfig, FaceAnalyzer,
        FaceEmbedder, AVAILABLE_MODELS, check_embedding_models
    )
except ImportError as e:
    print(f"Error importing core backend: {e}")
    print("Make sure core.py is in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedFaceGUI:
    """
    Main GUI Application for Face Processing System

    Provides a tabbed interface with the following features:
    1. System Overview - Monitor system status and statistics
    2. Download Faces - Download AI-generated faces or capture from camera
    3. Process & Embed - Create vector embeddings from face images
    4. Search Faces - Query similar faces using vector similarity or metadata
    5. Configuration - Manage system settings and database
    """

    def __init__(self):
        """Initialize the GUI application"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Processing System")
        self.root.geometry("1200x800")

        # Core system components
        self.system = None  # IntegratedFaceSystem instance
        self.download_thread = None  # Background download thread
        self.processing_thread = None  # Background processing thread

        # Application state
        self.is_downloading = False
        self.is_processing = False
        self.last_stats_update = 0

        # Create main container with scrollbar
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)

        # Create canvas and scrollbar for entire window
        self.main_canvas = tk.Canvas(self.main_container)
        self.main_scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)

        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar.pack(side="right", fill="y")

        # Bind canvas resize to expand scrollable frame width
        def on_canvas_configure(event):
            self.main_canvas.itemconfig(self.canvas_window, width=event.width)

        self.main_canvas.bind("<Configure>", on_canvas_configure)

        # Bind mousewheel to main canvas
        self._bind_main_mousewheel()

        # Create GUI
        self.create_widgets()
        self.setup_layout()

        # Initialize system
        self.initialize_system()

        # Start update loop
        self.update_display()

    # ============================================================================
    # GUI CREATION METHODS
    # ============================================================================

    def create_widgets(self):
        """
        Create all GUI widgets and tabs

        Creates a tabbed interface with five main tabs:
        - System Overview: Monitor status and statistics
        - Download Faces: Acquire face images
        - Process & Embed: Generate vector embeddings
        - Search Faces: Query the database
        - Configuration: System settings
        """
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.scrollable_frame)

        # Create the system menu
        self.create_system_menu()

        # Tab 1: System Overview - Status monitoring and statistics
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="System Overview")
        self.create_overview_tab()

        # Tab 2: Download - Face image acquisition
        self.download_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.download_frame, text="Download Faces")
        self.create_download_tab()

        # Tab 3: Process/Embed - Vector embedding generation
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text="Process & Embed")
        self.create_process_tab()

        # Tab 4: Search - Database queries
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Search Faces")
        self.create_search_tab()

        # Tab 5: Configuration - System settings
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.create_config_tab()

    def create_overview_tab(self):
        """Create system overview tab"""

        # System status frame
        status_frame = ttk.LabelFrame(self.overview_frame, text="System Status", padding=10)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Status labels
        self.status_labels = {}
        status_items = [
            ("Database Status", "db_status"),
            ("Total Faces", "total_faces"),
            ("Download Rate", "download_rate"),
            ("Processing Rate", "process_rate"),
            ("System Uptime", "uptime")
        ]

        for i, (label, key) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", padx=(0, 10))
            self.status_labels[key] = ttk.Label(status_frame, text="Initializing...")
            self.status_labels[key].grid(row=i, column=1, sticky="w")

        # Statistics frame
        stats_frame = ttk.LabelFrame(self.overview_frame, text="Statistics", padding=10)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Statistics text widget
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=70)
        self.stats_text.pack(fill="both", expand=True)

        # Control buttons
        control_frame = ttk.Frame(self.overview_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(control_frame, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Reset Statistics", command=self.reset_statistics).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Save Configuration", command=self.save_configuration).pack(side="left", padx=5)

    def create_download_tab(self):
        """Create download faces tab"""

        # Configure grid weights for proper resizing
        self.download_frame.columnconfigure(0, weight=1)
        self.download_frame.rowconfigure(3, weight=1)  # Preview frame expands

        # Download control frame
        control_frame = ttk.LabelFrame(self.download_frame, text="Download Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Download settings
        ttk.Label(control_frame, text="Download Source:").grid(row=0, column=0, sticky="w")
        self.download_source_var = tk.StringVar(value="thispersondoesnotexist")
        source_options = ["thispersondoesnotexist", "100k-faces"]
        source_combo = ttk.Combobox(control_frame, textvariable=self.download_source_var,
                                   values=source_options, width=25, state="readonly")
        source_combo.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Download Delay (seconds):").grid(row=1, column=0, sticky="w")
        self.download_delay_var = tk.DoubleVar(value=1.0)
        delay_spin = ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1,
                                textvariable=self.download_delay_var, width=10)
        delay_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Faces Directory:").grid(row=2, column=0, sticky="w")
        self.faces_dir_var = tk.StringVar(value="./faces")
        ttk.Entry(control_frame, textvariable=self.faces_dir_var, width=40).grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_faces_dir).grid(row=2, column=2, padx=5)

        # Download buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.download_button = ttk.Button(button_frame, text="Start Download", command=self.toggle_download)
        self.download_button.pack(side="left", padx=5)

        ttk.Button(button_frame, text="Download Single", command=self.download_single).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Take a Picture", command=self.take_picture_download).pack(side="left", padx=5)

        # Download Statistics Frame
        stats_frame = ttk.LabelFrame(self.download_frame, text="Download Statistics", padding=10)
        stats_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Create statistics labels
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill="x")

        self.download_stats_labels = {}
        stat_items = [
            ("Total Downloads:", "total_downloads"),
            ("Session Downloads:", "session_downloads"),
            ("Success Rate:", "success_rate"),
            ("Duplicates:", "duplicates"),
            ("Errors:", "errors"),
            ("Avg Speed:", "avg_speed"),
            ("Last Download:", "last_speed"),
            ("Download Rate:", "download_rate")
        ]

        for i, (label, key) in enumerate(stat_items):
            row = i // 4
            col = (i % 4) * 2
            ttk.Label(stats_grid, text=label, font=('TkDefaultFont', 9, 'bold')).grid(row=row, column=col, sticky="w", padx=(5, 2))
            self.download_stats_labels[key] = ttk.Label(stats_grid, text="0", font=('TkDefaultFont', 9))
            self.download_stats_labels[key].grid(row=row, column=col+1, sticky="w", padx=(0, 15))

        # Download status
        status_frame = ttk.LabelFrame(self.download_frame, text="Download Status", padding=10)
        status_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.download_status_text = scrolledtext.ScrolledText(status_frame, height=6, width=70)
        self.download_status_text.pack(fill="both", expand=True)

        # Download preview frame - thumbnails with scrolling
        preview_frame = ttk.LabelFrame(self.download_frame, text="Downloaded Images Preview", padding=10)
        preview_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # Create canvas with both horizontal and vertical scrollbars
        self.download_canvas = tk.Canvas(preview_frame, height=200)
        download_h_scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.download_canvas.xview)
        download_v_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.download_canvas.yview)
        self.download_thumbnails_frame = ttk.Frame(self.download_canvas)

        self.download_canvas.configure(
            xscrollcommand=download_h_scrollbar.set,
            yscrollcommand=download_v_scrollbar.set
        )

        # Grid layout for canvas and scrollbars
        self.download_canvas.grid(row=0, column=0, sticky="nsew")
        download_h_scrollbar.grid(row=1, column=0, sticky="ew")
        download_v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.download_canvas.create_window((0, 0), window=self.download_thumbnails_frame, anchor="nw")

        # Enable mousewheel scrolling for download canvas
        self._bind_mousewheel(self.download_canvas, self.download_thumbnails_frame)

        # Track thumbnails
        self.download_thumbnails = []
        self.download_thumbnail_refs = []  # Keep references to prevent garbage collection

    def create_process_tab(self):
        """Create process/embed tab"""

        # Configure grid weights for proper resizing
        self.process_frame.columnconfigure(0, weight=1)
        self.process_frame.rowconfigure(3, weight=1)  # Preview frame expands

        # Processing control frame
        control_frame = ttk.LabelFrame(self.process_frame, text="Processing Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Processing settings
        ttk.Label(control_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        self.batch_size_var = tk.IntVar(value=50)
        batch_spin = ttk.Spinbox(control_frame, from_=1, to=200, increment=1,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Max Workers:").grid(row=1, column=0, sticky="w")
        self.max_workers_var = tk.IntVar(value=4)
        workers_spin = ttk.Spinbox(control_frame, from_=1, to=8, increment=1,
                                  textvariable=self.max_workers_var, width=10)
        workers_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        # Processing buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.process_button = ttk.Button(button_frame, text="Process All Faces", command=self.start_processing)
        self.process_button.pack(side="left", padx=5)

        ttk.Button(button_frame, text="Process New Only", command=self.process_new_faces).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Stop Processing", command=self.stop_processing).pack(side="left", padx=5)

        # Embedding Statistics Frame
        embed_stats_frame = ttk.LabelFrame(self.process_frame, text="Embedding Statistics", padding=10)
        embed_stats_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Create statistics labels
        embed_stats_grid = ttk.Frame(embed_stats_frame)
        embed_stats_grid.pack(fill="x")

        self.embed_stats_labels = {}
        embed_stat_items = [
            ("Total Embedded:", "total_embeds"),
            ("Session Embeds:", "session_embeds"),
            ("Success Rate:", "success_rate"),
            ("Duplicates:", "duplicates"),
            ("Errors:", "errors"),
            ("Avg Speed:", "avg_speed"),
            ("Last Embed:", "last_speed"),
            ("Embed Rate:", "embed_rate")
        ]

        for i, (label, key) in enumerate(embed_stat_items):
            row = i // 4
            col = (i % 4) * 2
            ttk.Label(embed_stats_grid, text=label, font=('TkDefaultFont', 9, 'bold')).grid(row=row, column=col, sticky="w", padx=(5, 2))
            self.embed_stats_labels[key] = ttk.Label(embed_stats_grid, text="0", font=('TkDefaultFont', 9))
            self.embed_stats_labels[key].grid(row=row, column=col+1, sticky="w", padx=(0, 15))

        # Progress frame
        progress_frame = ttk.LabelFrame(self.process_frame, text="Processing Progress", padding=10)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.process_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.process_progress.pack(fill="x", pady=(0, 10))

        self.process_status_text = scrolledtext.ScrolledText(progress_frame, height=6, width=70)
        self.process_status_text.pack(fill="both", expand=True)

        # Processing preview frame - thumbnails with scrolling
        process_preview_frame = ttk.LabelFrame(self.process_frame, text="Embedding Images Preview", padding=10)
        process_preview_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        process_preview_frame.columnconfigure(0, weight=1)
        process_preview_frame.rowconfigure(0, weight=1)

        # Create canvas with both horizontal and vertical scrollbars
        self.process_canvas = tk.Canvas(process_preview_frame, height=200)
        process_h_scrollbar = ttk.Scrollbar(process_preview_frame, orient="horizontal", command=self.process_canvas.xview)
        process_v_scrollbar = ttk.Scrollbar(process_preview_frame, orient="vertical", command=self.process_canvas.yview)
        self.process_thumbnails_frame = ttk.Frame(self.process_canvas)

        self.process_canvas.configure(
            xscrollcommand=process_h_scrollbar.set,
            yscrollcommand=process_v_scrollbar.set
        )

        # Grid layout for canvas and scrollbars
        self.process_canvas.grid(row=0, column=0, sticky="nsew")
        process_h_scrollbar.grid(row=1, column=0, sticky="ew")
        process_v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.process_canvas.create_window((0, 0), window=self.process_thumbnails_frame, anchor="nw")

        # Enable mousewheel scrolling for process canvas
        self._bind_mousewheel(self.process_canvas, self.process_thumbnails_frame)

        # Track thumbnails
        self.process_thumbnails = []
        self.process_thumbnail_refs = []  # Keep references to prevent garbage collection

    def create_search_tab(self):
        """Create search faces tab"""

        # Configure grid weights for two-column layout
        self.search_frame.columnconfigure(0, weight=3)  # Controls take more space
        self.search_frame.columnconfigure(1, weight=1)  # Preview takes less space
        self.search_frame.rowconfigure(1, weight=1)      # Results expand

        # Search control frame (left side)
        control_frame = ttk.LabelFrame(self.search_frame, text="Search Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Search by image
        ttk.Label(control_frame, text="Search by Image:").grid(row=0, column=0, sticky="w")
        self.search_image_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.search_image_var, width=30).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_search_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Take a Picture", command=self.take_picture_search).grid(row=0, column=3, padx=5)

        # Number of results
        ttk.Label(control_frame, text="Number of Results:").grid(row=1, column=0, sticky="w")
        self.num_results_var = tk.IntVar(value=10)
        results_spin = ttk.Spinbox(control_frame, from_=1, to=50, increment=1,
                                  textvariable=self.num_results_var, width=10)
        results_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        # Search mode selection
        ttk.Label(control_frame, text="Search Mode:").grid(row=2, column=0, sticky="w")
        self.search_mode_var = tk.StringVar(value="vector")
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Radiobutton(mode_frame, text="Vector (Image)", variable=self.search_mode_var, value="vector").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Metadata", variable=self.search_mode_var, value="metadata").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Hybrid", variable=self.search_mode_var, value="hybrid").pack(side="left", padx=5)

        # Metadata filter frame
        metadata_frame = ttk.LabelFrame(control_frame, text="Metadata Filters (Optional)", padding=5)
        metadata_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        # Column 1 - Demographics
        demo_label = ttk.Label(metadata_frame, text="Demographics", font=('TkDefaultFont', 9, 'bold'))
        demo_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        # Sex filter
        ttk.Label(metadata_frame, text="Sex:").grid(row=1, column=0, sticky="w", padx=(10, 0))
        self.sex_filter_var = tk.StringVar(value="any")
        sex_combo = ttk.Combobox(metadata_frame, textvariable=self.sex_filter_var,
                                values=["any", "male", "female", "unknown"], width=15, state="readonly")
        sex_combo.grid(row=1, column=1, sticky="w", padx=(5, 10))

        # Age group filter
        ttk.Label(metadata_frame, text="Age Group:").grid(row=2, column=0, sticky="w", padx=(10, 0))
        self.age_filter_var = tk.StringVar(value="any")
        age_combo = ttk.Combobox(metadata_frame, textvariable=self.age_filter_var,
                                values=["any", "child", "young_adult", "adult", "middle_aged", "senior"],
                                width=15, state="readonly")
        age_combo.grid(row=2, column=1, sticky="w", padx=(5, 10))

        # Skin tone filter
        ttk.Label(metadata_frame, text="Skin Tone:").grid(row=3, column=0, sticky="w", padx=(10, 0))
        self.skin_tone_filter_var = tk.StringVar(value="any")
        skin_tone_combo = ttk.Combobox(metadata_frame, textvariable=self.skin_tone_filter_var,
                                      values=["any", "very_light", "light", "medium", "tan", "brown", "dark"],
                                      width=15, state="readonly")
        skin_tone_combo.grid(row=3, column=1, sticky="w", padx=(5, 10))

        # Skin color (broad category)
        ttk.Label(metadata_frame, text="Skin Color:").grid(row=4, column=0, sticky="w", padx=(10, 0))
        self.skin_color_filter_var = tk.StringVar(value="any")
        skin_color_combo = ttk.Combobox(metadata_frame, textvariable=self.skin_color_filter_var,
                                       values=["any", "light", "medium", "dark"],
                                       width=15, state="readonly")
        skin_color_combo.grid(row=4, column=1, sticky="w", padx=(5, 10))

        # Hair color filter
        ttk.Label(metadata_frame, text="Hair Color:").grid(row=5, column=0, sticky="w", padx=(10, 0))
        self.hair_color_filter_var = tk.StringVar(value="any")
        hair_color_combo = ttk.Combobox(metadata_frame, textvariable=self.hair_color_filter_var,
                                       values=["any", "black", "dark_brown", "brown", "blonde", "red", "gray", "light_gray", "other"],
                                       width=15, state="readonly")
        hair_color_combo.grid(row=5, column=1, sticky="w", padx=(5, 10))

        # Column 2 - Image Properties
        props_label = ttk.Label(metadata_frame, text="Image Properties", font=('TkDefaultFont', 9, 'bold'))
        props_label.grid(row=0, column=2, columnspan=2, sticky="w", pady=(0, 5))

        # Brightness filter
        ttk.Label(metadata_frame, text="Brightness:").grid(row=1, column=2, sticky="w", padx=(10, 0))
        self.brightness_filter_var = tk.StringVar(value="any")
        brightness_combo = ttk.Combobox(metadata_frame, textvariable=self.brightness_filter_var,
                                       values=["any", "bright", "dark"], width=15, state="readonly")
        brightness_combo.grid(row=1, column=3, sticky="w", padx=(5, 0))

        # Quality filter
        ttk.Label(metadata_frame, text="Quality:").grid(row=2, column=2, sticky="w", padx=(10, 0))
        self.quality_filter_var = tk.StringVar(value="any")
        quality_combo = ttk.Combobox(metadata_frame, textvariable=self.quality_filter_var,
                                    values=["any", "high", "medium"], width=15, state="readonly")
        quality_combo.grid(row=2, column=3, sticky="w", padx=(5, 0))

        # Face detection filter
        ttk.Label(metadata_frame, text="Has Face:").grid(row=3, column=2, sticky="w", padx=(10, 0))
        self.has_face_var = tk.StringVar(value="any")
        face_combo = ttk.Combobox(metadata_frame, textvariable=self.has_face_var,
                                 values=["any", "yes", "no"], width=15, state="readonly")
        face_combo.grid(row=3, column=3, sticky="w", padx=(5, 0))

        # Search button
        ttk.Button(control_frame, text="Search Faces", command=self.search_faces).grid(row=4, column=0, columnspan=3, pady=10)

        # Query image preview frame (right side)
        query_preview_frame = ttk.LabelFrame(self.search_frame, text="Query Image Preview", padding=10)
        query_preview_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=5, pady=5)

        # Preview label for query image
        self.query_preview_label = ttk.Label(query_preview_frame, text="No image selected",
                                             relief="solid", borderwidth=1,
                                             anchor="center", background="lightgray")
        self.query_preview_label.pack(fill="both", expand=True, padx=5, pady=5)
        self.query_preview_photo = None  # Keep reference to prevent garbage collection

        # Results frame (spans both columns)
        self.results_frame = ttk.LabelFrame(self.search_frame, text="Search Results", padding=10)
        self.results_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Results display
        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_frame_inner = ttk.Frame(self.results_canvas)

        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")
        self.results_canvas.create_window((0, 0), window=self.results_frame_inner, anchor="nw")

    def create_config_tab(self):
        """Create configuration tab"""

        # Database config frame
        db_frame = ttk.LabelFrame(self.config_frame, text="PostgreSQL Database Configuration", padding=10)
        db_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # PostgreSQL Configuration
        ttk.Label(db_frame, text="Host:").grid(row=0, column=0, sticky="w", pady=2)
        self.pg_host_var = tk.StringVar(value="localhost")
        ttk.Entry(db_frame, textvariable=self.pg_host_var, width=30).grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(db_frame, text="Port:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.pg_port_var = tk.StringVar(value="5432")
        ttk.Entry(db_frame, textvariable=self.pg_port_var, width=10).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Label(db_frame, text="Database:").grid(row=1, column=0, sticky="w", pady=2)
        self.pg_db_var = tk.StringVar(value="vector_db")
        ttk.Entry(db_frame, textvariable=self.pg_db_var, width=30).grid(row=1, column=1, sticky="w", padx=(5, 0))

        ttk.Label(db_frame, text="User:").grid(row=1, column=2, sticky="w", padx=(10, 0))
        self.pg_user_var = tk.StringVar(value="postgres")
        ttk.Entry(db_frame, textvariable=self.pg_user_var, width=10).grid(row=1, column=3, sticky="w", padx=(5, 0))

        ttk.Label(db_frame, text="Password:").grid(row=2, column=0, sticky="w", pady=2)
        self.pg_password_var = tk.StringVar(value="postgres")
        ttk.Entry(db_frame, textvariable=self.pg_password_var, width=30, show="*").grid(row=2, column=1, sticky="w", padx=(5, 0))

        ttk.Button(db_frame, text="Test Connection",
                  command=self.test_pg_connection).grid(row=2, column=2, columnspan=2, padx=(10, 0))

        # Embedding Model Configuration
        embedding_frame = ttk.LabelFrame(self.config_frame, text="Embedding Model Configuration", padding=10)
        embedding_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(embedding_frame, text="Embedding Model:").grid(row=0, column=0, sticky="w")
        self.embedding_model_var = tk.StringVar(value="statistical")

        # Create dropdown with all models
        model_options = ["statistical", "facenet", "arcface", "deepface", "vggface2", "openface"]
        embedding_combo = ttk.Combobox(embedding_frame, textvariable=self.embedding_model_var,
                                      values=model_options, width=20, state="readonly")
        embedding_combo.grid(row=0, column=1, sticky="w", padx=(5, 10))

        ttk.Button(embedding_frame, text="Check Model Availability",
                  command=self.check_embedding_models).grid(row=0, column=2, padx=5)

        # Bind model change event
        self.embedding_model_var.trace('w', self.on_embedding_model_changed)

        # Model change warning label
        self.model_warning_label = ttk.Label(embedding_frame, text="", foreground="red", font=('TkDefaultFont', 9, 'bold'))
        self.model_warning_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # Model description
        model_desc_frame = ttk.Frame(embedding_frame)
        model_desc_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        self.model_info_text = scrolledtext.ScrolledText(model_desc_frame, height=8, width=70, wrap=tk.WORD)
        self.model_info_text.pack(fill="both", expand=True)

        # Insert model descriptions
        model_descriptions = """
Embedding Models:

• Statistical (Default): Simple statistical features. Always available. Fast but lower accuracy.
  Size: 512 dimensions

• FaceNet: Deep learning model. High accuracy for face recognition.
  Install: pip install facenet-pytorch torch torchvision
  Size: 512 dimensions

• ArcFace: State-of-the-art face recognition. Best accuracy.
  Install: pip install insightface onnxruntime
  Size: 512 dimensions

• DeepFace: Multi-purpose deep learning. Good general performance.
  Install: pip install deepface
  Size: 4096 dimensions

• VGGFace2: Deep CNN model. Good accuracy, slower.
  Install: pip install deepface
  Size: 2622 dimensions

• OpenFace: Lightweight deep learning. Fast and reasonable accuracy.
  Install: pip install deepface
  Size: 128 dimensions
"""
        self.model_info_text.insert('1.0', model_descriptions)
        self.model_info_text.config(state='disabled')

        # System config frame
        system_frame = ttk.LabelFrame(self.config_frame, text="System Configuration", padding=10)
        system_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # Dependencies status
        deps_frame = ttk.LabelFrame(self.config_frame, text="Dependencies Status", padding=10)
        deps_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.deps_text = scrolledtext.ScrolledText(deps_frame, height=10, width=70)
        self.deps_text.pack(fill="both", expand=True)

        # Config buttons
        button_frame = ttk.Frame(self.config_frame)
        button_frame.grid(row=4, column=0, pady=10)

        ttk.Button(button_frame, text="Initialize Vector Database", command=self.initialize_vector_database).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Initialize Download Directory", command=self.initialize_download_directory).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Re-embed All Data", command=self.reembed_all_data).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Check Dependencies", command=self.check_dependencies).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load Configuration", command=self.load_configuration).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Configuration", command=self.save_configuration).pack(side="left", padx=5)

    def setup_layout(self):
        """Setup the main layout"""
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

    def create_system_menu(self):
        """Create the main system menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=system_menu)

        system_menu.add_command(label="Validate and Fix Metadata", command=self.validate_metadata_files)
        system_menu.add_separator()
        system_menu.add_command(label="Exit", command=self.root.quit)

    def validate_metadata_files(self):
        """Validate and fix metadata JSON files."""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        faces_dir = self.faces_dir_var.get()
        if not os.path.isdir(faces_dir):
            messagebox.showerror("Error", f"Faces directory not found: {faces_dir}")
            return

        self.log_message("Starting metadata validation...")
        
        # Run validation in a separate thread to keep UI responsive
        threading.Thread(target=self._run_metadata_validation, daemon=True).start()

    def _run_metadata_validation(self):
        """The actual validation logic running in a background thread."""
        faces_dir = self.faces_dir_var.get()
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(faces_dir).rglob(ext))

        missing_json = []
        invalid_json = []
        
        required_keys = ['filename', 'face_id', 'md5_hash', 'image_properties', 'face_features']

        for image_path in image_files:
            json_path = image_path.with_suffix('.json')
            if not json_path.exists():
                missing_json.append(image_path)
            else:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if not all(key in data for key in required_keys):
                        invalid_json.append(image_path)
                except (json.JSONDecodeError, TypeError):
                    invalid_json.append(image_path)

        # Update UI from the main thread
        self.root.after(0, self._show_validation_results, missing_json, invalid_json)

    def _show_validation_results(self, missing_json, invalid_json):
        """Display validation results in the UI."""
        total_issues = len(missing_json) + len(invalid_json)

        if total_issues == 0:
            self.log_message("Metadata validation complete. All files are well-defined.")
            messagebox.showinfo("Validation Complete", "All metadata files are present and well-formed.")
            return

        report = f"Metadata validation found {total_issues} issues:\n\n"
        if missing_json:
            report += f"Images with missing JSON files: {len(missing_json)}\n"
            for path in missing_json[:5]:
                report += f"  - {path.name}\n"
            if len(missing_json) > 5:
                report += "  - ...and more\n"

        if invalid_json:
            report += f"\nImages with invalid or incomplete JSON files: {len(invalid_json)}\n"
            for path in invalid_json[:5]:
                report += f"  - {path.name}\n"
            if len(invalid_json) > 5:
                report += "  - ...and more\n"

        self.log_message(report, "error")

        response = messagebox.askyesno(
            "Fix Metadata Issues?",
            f"{report}\nDo you want to attempt to fix these issues by regenerating the metadata?"
        )

        if response:
            self.fix_metadata_files(missing_json + invalid_json)

    def fix_metadata_files(self, files_to_fix):
        """Regenerate metadata for the given list of image files."""
        self.log_message(f"Starting metadata fix for {len(files_to_fix)} files...")
        
        # Run fix in a background thread
        threading.Thread(target=self._run_metadata_fix, args=(files_to_fix,), daemon=True).start()

    def _run_metadata_fix(self, files_to_fix):
        """The actual metadata fixing logic."""
        fixed_count = 0
        error_count = 0

        for image_path in files_to_fix:
            try:
                self.root.after(0, self.log_message, f"Fixing metadata for {image_path.name}...")
                
                analyzer = FaceAnalyzer()
                features = analyzer.analyze_face(str(image_path))

                with Image.open(image_path) as img:
                    image_width, image_height = img.size
                    image_format = img.format
                    image_mode = img.mode

                with open(image_path, 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                metadata = {
                    'filename': image_path.name,
                    'file_path': str(image_path),
                    'face_id': f"fixed_{timestamp}",
                    'md5_hash': image_hash,
                    'download_timestamp': datetime.now().isoformat(),
                    'source_url': 'local_fix',
                    'file_size_bytes': image_path.stat().st_size,
                    'image_properties': {
                        'width': image_width,
                        'height': image_height,
                        'format': image_format,
                        'mode': image_mode,
                    },
                    'face_features': features,
                    'queryable_attributes': {
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
                }

                json_path = image_path.with_suffix('.json')
                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=2, default=self._json_numpy_fallback)
                
                fixed_count += 1
            except Exception as e:
                self.root.after(0, self.log_message, f"Error fixing {image_path.name}: {e}", "error")
                error_count += 1

        self.root.after(0, self._show_fix_results, fixed_count, error_count)

    def _show_fix_results(self, fixed_count, error_count):
        """Display the results of the metadata fix operation."""
        self.log_message(f"Metadata fix complete. Fixed: {fixed_count}, Errors: {error_count}")
        messagebox.showinfo("Fix Complete", f"Successfully fixed {fixed_count} metadata files.\nEncountered {error_count} errors.")

    def _json_numpy_fallback(self, obj):
        """Fallback for JSON serialization of numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def _json_numpy_fallback(self, obj):
        """Fallback for JSON serialization of numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

        # Create top-level menu
        self.create_system_menu()

    def create_system_menu(self):
        """Create the main system menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=system_menu)

        system_menu.add_command(label="Validate and Fix Metadata", command=self.validate_metadata_files)
        system_menu.add_separator()
        system_menu.add_command(label="Exit", command=self.root.quit)

    def validate_metadata_files(self):
        """Validate and fix metadata JSON files."""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        faces_dir = self.faces_dir_var.get()
        if not os.path.isdir(faces_dir):
            messagebox.showerror("Error", f"Faces directory not found: {faces_dir}")
            return

        self.log_message("Starting metadata validation...")
        
        # Run validation in a separate thread to keep UI responsive
        threading.Thread(target=self._run_metadata_validation, daemon=True).start()

    def _run_metadata_validation(self):
        """The actual validation logic running in a background thread."""
        faces_dir = self.faces_dir_var.get()
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(faces_dir).rglob(ext))

        missing_json = []
        invalid_json = []
        
        required_keys = ['filename', 'face_id', 'md5_hash', 'image_properties', 'face_features']

        for image_path in image_files:
            json_path = image_path.with_suffix('.json')
            if not json_path.exists():
                missing_json.append(image_path)
            else:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    if not all(key in data for key in required_keys):
                        invalid_json.append(image_path)
                except (json.JSONDecodeError, TypeError):
                    invalid_json.append(image_path)

        # Update UI from the main thread
        self.root.after(0, self._show_validation_results, missing_json, invalid_json)

    def _show_validation_results(self, missing_json, invalid_json):
        """Display validation results in the UI."""
        total_issues = len(missing_json) + len(invalid_json)

        if total_issues == 0:
            self.log_message("Metadata validation complete. All files are well-defined.")
            messagebox.showinfo("Validation Complete", "All metadata files are present and well-formed.")
            return

        report = f"Metadata validation found {total_issues} issues:\n\n"
        if missing_json:
            report += f"Images with missing JSON files: {len(missing_json)}\n"
            for path in missing_json[:5]:
                report += f"  - {path.name}\n"
            if len(missing_json) > 5:
                report += "  - ...and more\n"

        if invalid_json:
            report += f"\nImages with invalid or incomplete JSON files: {len(invalid_json)}\n"
            for path in invalid_json[:5]:
                report += f"  - {path.name}\n"
            if len(invalid_json) > 5:
                report += "  - ...and more\n"

        self.log_message(report, "error")

        response = messagebox.askyesno(
            "Fix Metadata Issues?",
            f"{report}\nDo you want to attempt to fix these issues by regenerating the metadata?"
        )

        if response:
            self.fix_metadata_files(missing_json + invalid_json)

    def fix_metadata_files(self, files_to_fix):
        """Regenerate metadata for the given list of image files."""
        self.log_message(f"Starting metadata fix for {len(files_to_fix)} files...")
        
        # Run fix in a background thread
        threading.Thread(target=self._run_metadata_fix, args=(files_to_fix,), daemon=True).start()

    def _run_metadata_fix(self, files_to_fix):
        """The actual metadata fixing logic."""
        fixed_count = 0
        error_count = 0

        for image_path in files_to_fix:
            try:
                self.root.after(0, self.log_message, f"Fixing metadata for {image_path.name}...")
                
                analyzer = FaceAnalyzer()
                features = analyzer.analyze_face(str(image_path))

                with Image.open(image_path) as img:
                    image_width, image_height = img.size
                    image_format = img.format
                    image_mode = img.mode

                with open(image_path, 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                metadata = {
                    'filename': image_path.name,
                    'file_path': str(image_path),
                    'face_id': f"fixed_{timestamp}",
                    'md5_hash': image_hash,
                    'download_timestamp': datetime.now().isoformat(),
                    'source_url': 'local_fix',
                    'file_size_bytes': image_path.stat().st_size,
                    'image_properties': {
                        'width': image_width,
                        'height': image_height,
                        'format': image_format,
                        'mode': image_mode,
                    },
                    'face_features': features,
                    'queryable_attributes': {
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
                }

                json_path = image_path.with_suffix('.json')
                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=2, default=self._json_numpy_fallback)
                
                fixed_count += 1
            except Exception as e:
                self.root.after(0, self.log_message, f"Error fixing {image_path.name}: {e}", "error")
                error_count += 1

        self.root.after(0, self._show_fix_results, fixed_count, error_count)

    def _show_fix_results(self, fixed_count, error_count):
        """Display the results of the metadata fix operation."""
        self.log_message(f"Metadata fix complete. Fixed: {fixed_count}, Errors: {error_count}")
        messagebox.showinfo("Fix Complete", f"Successfully fixed {fixed_count} metadata files.\nEncountered {error_count} errors.")

    def _json_numpy_fallback(self, obj):
        """Fallback for JSON serialization of numpy types"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

        # Configure column weights for responsive design
        for i in range(5):  # Number of tabs
            self.root.grid_columnconfigure(i, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def initialize_system(self):
        """Initialize the face processing system"""
        try:
            self.system = IntegratedFaceSystem()
            if self.system.initialize():
                self.log_message("System initialized successfully")
                self.update_configuration_from_system()
                # Check for model mismatch after initialization
                self.check_model_mismatch_on_startup()
            else:
                self.log_message("Failed to initialize system", "error")
                messagebox.showerror("Error", "Failed to initialize system. Check dependencies.")
        except Exception as e:
            self.log_message(f"Error initializing system: {e}", "error")
            messagebox.showerror("Error", f"Error initializing system: {e}")

    def update_configuration_from_system(self):
        """Update GUI configuration from system"""
        if self.system:
            config = self.system.config
            self.faces_dir_var.set(config.faces_dir)

            # Load PostgreSQL database settings
            self.pg_host_var.set(getattr(config, 'db_host', 'localhost'))
            self.pg_port_var.set(str(getattr(config, 'db_port', 5432)))
            self.pg_db_var.set(getattr(config, 'db_name', 'vector_db'))
            self.pg_user_var.set(getattr(config, 'db_user', 'postgres'))
            self.pg_password_var.set(getattr(config, 'db_password', ''))

            self.download_delay_var.set(config.download_delay)
            self.batch_size_var.set(config.batch_size)
            self.max_workers_var.set(config.max_workers)
            self.embedding_model_var.set(config.embedding_model)
            self.download_source_var.set(config.download_source)

    def log_message(self, message: str, level: str = "info"):
        """Log message to appropriate text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Log to download status if downloading
        if hasattr(self, 'download_status_text'):
            self.download_status_text.insert(tk.END, formatted_message)
            self.download_status_text.see(tk.END)

        # Log to process status if processing
        if hasattr(self, 'process_status_text'):
            self.process_status_text.insert(tk.END, formatted_message)
            self.process_status_text.see(tk.END)

        # Log to logger
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def _bind_main_mousewheel(self):
        """Bind mousewheel scrolling to main canvas"""
        def on_mousewheel(event):
            # Scroll vertically
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mousewheel to main canvas
        self.main_canvas.bind_all("<MouseWheel>", on_mousewheel)

    def _bind_mousewheel(self, canvas, frame):
        """Bind mousewheel scrolling to canvas"""
        def on_mousewheel(event):
            # Scroll vertically
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_enter(event):
            # Bind mousewheel when mouse enters the canvas
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def on_leave(event):
            # Unbind mousewheel when mouse leaves the canvas
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)

        # Update scroll region when frame size changes
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", on_frame_configure)

    def initialize_vector_database(self):
        """Initialize vector database"""
        try:
            if self.system:
                # Reinitialize database with current configuration
                if self.system.db_manager.initialize():
                    self.log_message(f"PostgreSQL database initialized successfully")
                    messagebox.showinfo("Success", "PostgreSQL database initialized successfully!")
                else:
                    self.log_message("Failed to initialize PostgreSQL database", "error")
                    messagebox.showerror("Error", "Failed to initialize PostgreSQL database. Check connection and configuration.")
            else:
                messagebox.showerror("Error", "System not initialized")
        except Exception as e:
            self.log_message(f"Error initializing vector database: {e}", "error")
            messagebox.showerror("Error", f"Error initializing vector database: {e}")

    def initialize_download_directory(self):
        """Initialize download directory"""
        try:
            # Get directory path from GUI
            download_dir = self.faces_dir_var.get()

            # Create directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)

            # Update system config if system exists
            if self.system:
                self.system.config.faces_dir = download_dir
                self.system.downloader.config.faces_dir = download_dir

            # Count existing files
            existing_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                existing_files.extend(Path(download_dir).rglob(ext))

            file_count = len(existing_files)

            # Calculate directory size
            total_size = sum(f.stat().st_size for f in existing_files if f.is_file())
            size_mb = total_size / (1024 * 1024)

            self.log_message(f"Download directory initialized: {download_dir}")
            self.log_message(f"Existing files: {file_count} ({size_mb:.2f} MB)")

            messagebox.showinfo("Success",
                f"Download directory initialized successfully!\n\n"
                f"Path: {download_dir}\n"
                f"Existing files: {file_count}\n"
                f"Total size: {size_mb:.2f} MB")

        except Exception as e:
            self.log_message(f"Error initializing download directory: {e}", "error")
            messagebox.showerror("Error", f"Error initializing download directory: {e}")

    def update_display(self):
        """Update display elements"""
        try:
            if self.system:
                status = self.system.get_system_status()

                # Update status labels
                if hasattr(self, 'status_labels'):
                    db_info = status.get('database', {})
                    stats = status.get('statistics', {})

                    self.status_labels['db_status'].config(text="Connected" if db_info else "Disconnected")
                    self.status_labels['total_faces'].config(text=str(db_info.get('count', 0)))
                    self.status_labels['download_rate'].config(text=f"{stats.get('download_rate', 0):.2f}/sec")
                    self.status_labels['process_rate'].config(text=f"{stats.get('embed_rate', 0):.2f}/sec")
                    self.status_labels['uptime'].config(text=f"{stats.get('elapsed_time', 0):.0f}s")

                # Update download statistics display
                if hasattr(self, 'download_stats_labels'):
                    stats = status.get('statistics', {})
                    self.download_stats_labels['total_downloads'].config(text=f"{stats.get('download_success', 0)}")
                    self.download_stats_labels['session_downloads'].config(text=f"{stats.get('session_downloads', 0)}")
                    self.download_stats_labels['success_rate'].config(text=f"{stats.get('download_success_rate', 0):.1f}%")
                    self.download_stats_labels['duplicates'].config(text=f"{stats.get('download_duplicates', 0)}")
                    self.download_stats_labels['errors'].config(text=f"{stats.get('download_errors', 0)}")
                    self.download_stats_labels['avg_speed'].config(text=f"{stats.get('avg_download_time', 0):.2f}s")
                    self.download_stats_labels['last_speed'].config(text=f"{stats.get('last_download_time', 0):.2f}s")
                    self.download_stats_labels['download_rate'].config(text=f"{stats.get('download_rate', 0):.3f}/s")

                # Update embedding statistics display
                if hasattr(self, 'embed_stats_labels'):
                    stats = status.get('statistics', {})
                    self.embed_stats_labels['total_embeds'].config(text=f"{stats.get('embed_success', 0)}")
                    self.embed_stats_labels['session_embeds'].config(text=f"{stats.get('session_embeds', 0)}")
                    self.embed_stats_labels['success_rate'].config(text=f"{stats.get('embed_success_rate', 0):.1f}%")
                    self.embed_stats_labels['duplicates'].config(text=f"{stats.get('embed_duplicates', 0)}")
                    self.embed_stats_labels['errors'].config(text=f"{stats.get('embed_errors', 0)}")
                    self.embed_stats_labels['avg_speed'].config(text=f"{stats.get('avg_embed_time', 0):.2f}s")
                    self.embed_stats_labels['last_speed'].config(text=f"{stats.get('last_embed_time', 0):.2f}s")
                    self.embed_stats_labels['embed_rate'].config(text=f"{stats.get('embed_rate', 0):.3f}/s")

                # Update statistics text (throttled)
                current_time = time.time()
                if current_time - self.last_stats_update > 2.0:  # Update every 2 seconds
                    if hasattr(self, 'stats_text'):
                        self.stats_text.delete(1.0, tk.END)
                        stats_str = json.dumps(status, indent=2)
                        self.stats_text.insert(1.0, stats_str)
                    self.last_stats_update = current_time

        except Exception as e:
            pass  # Ignore update errors

        # Schedule next update
        self.root.after(1000, self.update_display)

    # ============================================================================
    # DOWNLOAD MANAGEMENT METHODS
    # ============================================================================

    def toggle_download(self):
        """Toggle download process"""
        if not self.is_downloading:
            self.start_download()
        else:
            self.stop_download()

    def start_download(self):
        """Start downloading faces"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            # Update configuration
            self.system.config.download_delay = self.download_delay_var.get()
            self.system.config.faces_dir = self.faces_dir_var.get()
            self.system.config.download_source = self.download_source_var.get()

            # Create faces directory
            os.makedirs(self.system.config.faces_dir, exist_ok=True)

            # Start download
            self.download_thread = self.system.downloader.start_download_loop(
                callback=self.on_face_downloaded
            )

            self.is_downloading = True
            self.download_button.config(text="Stop Download")
            self.log_message("Download started")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start download: {e}")

    def stop_download(self):
        """Stop downloading faces"""
        if self.system:
            self.system.downloader.stop_download_loop()

        self.is_downloading = False
        self.download_button.config(text="Start Download")
        self.log_message("Download stopped")

    def download_single(self):
        """Download a single face"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            file_path = self.system.downloader.download_face()
            if file_path:
                self.log_message(f"Downloaded: {os.path.basename(file_path)}")
            else:
                self.log_message("No new face downloaded (duplicate or error)")
        except Exception as e:
            self.log_message(f"Download error: {e}", "error")

    def take_picture_download(self):
        """Take a picture from camera and save to download directory"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            import cv2
        except ImportError:
            messagebox.showerror("Error", "OpenCV (cv2) is required for camera capture.\nInstall with: pip install opencv-python")
            return

        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera. Please check camera connection.")
                return

            self.log_message("Opening camera... Press SPACE to capture, ESC to cancel")

            camera_window_name = "Camera - Press SPACE to Capture, ESC to Cancel"
            captured_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to read from camera")
                    break

                # Display the frame
                cv2.imshow(camera_window_name, frame)

                # Wait for key press
                key = cv2.waitKey(1) & 0xFF

                # SPACE key to capture
                if key == 32:  # SPACE
                    captured_frame = frame.copy()
                    self.log_message("Picture captured!")
                    break

                # ESC key to cancel
                elif key == 27:  # ESC
                    self.log_message("Camera capture cancelled")
                    break

            # Release camera and close window
            cap.release()
            cv2.destroyAllWindows()

            # Save captured image if available
            if captured_frame is not None:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"camera_capture_{timestamp}.jpg"
                file_path = os.path.join(self.system.config.faces_dir, filename)

                # Save image
                cv2.imwrite(file_path, captured_frame)
                self.log_message(f"Image saved: {filename}")

                # Add thumbnail to preview
                self.add_download_thumbnail(file_path)

                # Process for embedding if needed
                messagebox.showinfo("Success", f"Picture captured and saved!\n\n{filename}")

        except Exception as e:
            self.log_message(f"Camera capture error: {e}", "error")
            messagebox.showerror("Error", f"Failed to capture picture: {e}")

    def take_picture_search(self):
        """Take a picture from camera and use for search"""
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Error", "OpenCV (cv2) is required for camera capture.\nInstall with: pip install opencv-python")
            return

        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera. Please check camera connection.")
                return

            self.log_message("Opening camera for search... Press SPACE to capture, ESC to cancel")

            camera_window_name = "Camera - Press SPACE to Capture, ESC to Cancel"
            captured_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to read from camera")
                    break

                # Display the frame
                cv2.imshow(camera_window_name, frame)

                # Wait for key press
                key = cv2.waitKey(1) & 0xFF

                # SPACE key to capture
                if key == 32:  # SPACE
                    captured_frame = frame.copy()
                    self.log_message("Picture captured for search!")
                    break

                # ESC key to cancel
                elif key == 27:  # ESC
                    self.log_message("Camera capture cancelled")
                    break

            # Release camera and close window
            cap.release()
            cv2.destroyAllWindows()

            # Save captured image if available
            if captured_frame is not None:
                # Create temp directory if it doesn't exist
                temp_dir = os.path.join(self.system.config.faces_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"search_query_{timestamp}.jpg"
                file_path = os.path.join(temp_dir, filename)

                # Save image
                cv2.imwrite(file_path, captured_frame)
                self.log_message(f"Search image saved: {filename}")

                # Set the path to search image variable
                self.search_image_var.set(file_path)

                # Update query preview
                self.update_query_preview(file_path)

                messagebox.showinfo("Success", f"Picture captured for search!\n\nYou can now click 'Search Faces' to find similar faces.")

        except Exception as e:
            self.log_message(f"Camera capture error: {e}", "error")
            messagebox.showerror("Error", f"Failed to capture picture: {e}")

    def on_face_downloaded(self, file_path: str):
        """Callback when a face is downloaded"""
        # Use root.after to schedule GUI updates on the main thread
        self.root.after(0, lambda: self._update_download_ui(file_path))

    def _update_download_ui(self, file_path: str):
        """Update download UI on main thread"""
        self.log_message(f"Downloaded: {os.path.basename(file_path)}")
        # Add thumbnail to preview
        self.add_download_thumbnail(file_path)

    def add_download_thumbnail(self, file_path: str):
        """Add thumbnail to download preview"""
        try:
            # Load and resize image
            image = Image.open(file_path)

            # Get original dimensions
            orig_width, orig_height = image.size

            image.thumbnail((120, 120), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Create thumbnail frame with border
            thumb_frame = ttk.Frame(self.download_thumbnails_frame, relief="solid", borderwidth=1)
            thumb_frame.pack(side="left", padx=5, pady=5)

            # Display image
            image_label = ttk.Label(thumb_frame, image=photo)
            image_label.pack()

            # Get file size
            file_size = os.path.getsize(file_path) / 1024  # KB

            # Display info below image
            filename = os.path.basename(file_path)
            info_text = f"{filename}\n{orig_width}x{orig_height} | {file_size:.0f}KB"

            name_label = ttk.Label(thumb_frame, text=info_text,
                                  wraplength=120, font=('TkDefaultFont', 7),
                                  justify="center")
            name_label.pack()

            # Keep references
            self.download_thumbnail_refs.append(photo)
            self.download_thumbnails.append(thumb_frame)

            # Limit to last 20 thumbnails
            if len(self.download_thumbnails) > 20:
                old_frame = self.download_thumbnails.pop(0)
                old_frame.destroy()
                self.download_thumbnail_refs.pop(0)

            # Update scroll region
            self.download_thumbnails_frame.update_idletasks()
            self.download_canvas.configure(scrollregion=self.download_canvas.bbox("all"))
            # Auto-scroll to the right to show latest
            self.download_canvas.xview_moveto(1.0)

        except Exception as e:
            logger.error(f"Error adding download thumbnail: {e}")

    # ============================================================================
    # FACE PROCESSING METHODS
    # ============================================================================

    def start_processing(self):
        """Start processing all faces"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return

        # Get file counts
        try:
            from pathlib import Path
            all_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_files.extend(Path(self.system.config.faces_dir).rglob(ext))

            new_files = self.system.processor.get_new_files_only()
            existing_count = len(all_files) - len(new_files)

            if len(all_files) == 0:
                messagebox.showinfo("No Files",
                    "No image files found in the faces directory!\n\n"
                    f"Directory: {self.system.config.faces_dir}")
                return

            # Show confirmation with details
            response = messagebox.askokcancel(
                "Process All Faces",
                f"Process ALL faces in directory:\n\n"
                f"Total files: {len(all_files)}\n"
                f"  • New files: {len(new_files)} (will be processed)\n"
                f"  • Already in DB: {existing_count} (will be skipped)\n\n"
                f"Using '{self.system.config.embedding_model}' model\n\n"
                f"Duplicates will be automatically skipped.\n\n"
                f"Continue?"
            )

            if not response:
                return

        except Exception as e:
            self.log_message(f"Error checking files: {e}", "error")

        self.is_processing = True
        self.process_button.config(state="disabled")
        self.process_progress.start()

        def process_worker():
            try:
                self.system.processor.process_all_faces(
                    callback=self.on_face_processed
                )
                self.log_message("Processing completed")
            except Exception as e:
                self.log_message(f"Processing error: {e}", "error")
            finally:
                self.is_processing = False
                self.root.after(0, lambda: [
                    self.process_button.config(state="normal"),
                    self.process_progress.stop()
                ])

        self.processing_thread = threading.Thread(target=process_worker, daemon=True)
        self.processing_thread.start()

    def process_new_faces(self):
        """Process only new faces (not in database)"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return

        # Get count of new files first
        try:
            new_files = self.system.processor.get_new_files_only()

            if len(new_files) == 0:
                messagebox.showinfo("No New Files",
                    "All files in the faces directory have already been processed!\n\n"
                    "No new faces to embed.")
                return

            # Ask for confirmation
            response = messagebox.askokcancel(
                "Process New Files Only",
                f"Found {len(new_files)} NEW files that haven't been processed yet.\n\n"
                f"These files will be:\n"
                f"1. Analyzed for facial features\n"
                f"2. Embedded using '{self.system.config.embedding_model}' model\n"
                f"3. Added to the database\n\n"
                f"Already processed files will be skipped.\n\n"
                f"Continue?"
            )

            if not response:
                return

            self.is_processing = True
            self.process_button.config(state="disabled")
            self.process_progress.start()

            def process_worker():
                try:
                    self.log_message(f"Processing {len(new_files)} new files only...")
                    result_stats = self.system.processor.process_new_faces_only(
                        callback=self.on_face_processed
                    )
                    self.log_message(f"New files processing completed: {result_stats['processed']} processed, {result_stats['errors']} errors")
                except Exception as e:
                    self.log_message(f"Processing error: {e}", "error")
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: [
                        self.process_button.config(state="normal"),
                        self.process_progress.stop()
                    ])

            self.processing_thread = threading.Thread(target=process_worker, daemon=True)
            self.processing_thread.start()

        except Exception as e:
            self.log_message(f"Error checking new files: {e}", "error")
            messagebox.showerror("Error", f"Failed to check for new files: {e}")

    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        self.process_button.config(state="normal")
        self.process_progress.stop()
        self.log_message("Processing stopped")

    def on_face_processed(self, face_data):
        """Callback when a face is processed"""
        # Use root.after to schedule GUI updates on the main thread
        self.root.after(0, lambda: self._update_process_ui(face_data))

    def _update_process_ui(self, face_data):
        """Update process UI on main thread"""
        # Display detailed file information
        file_path = face_data.file_path
        filename = os.path.basename(file_path)
        features = face_data.features

        # Format detailed log message
        log_msg = f"✅ Processed: {filename}\n"
        log_msg += f"   📁 Size: {features.get('size_bytes', 0) / 1024:.1f} KB\n"
        log_msg += f"   📐 Dimensions: {features.get('width', 'N/A')}x{features.get('height', 'N/A')}\n"
        log_msg += f"   🎨 Format: {features.get('format', 'N/A')}\n"

        if 'brightness' in features:
            log_msg += f"   💡 Brightness: {features.get('brightness', 0):.1f}\n"
        if 'contrast' in features:
            log_msg += f"   🔆 Contrast: {features.get('contrast', 0):.1f}\n"
        if 'faces_detected' in features:
            log_msg += f"   👤 Faces Detected: {features.get('faces_detected', 0)}\n"

        log_msg += f"   🔑 Hash: {face_data.image_hash[:12]}...\n"
        log_msg += f"   🧬 Embedding: {len(face_data.embedding)} dimensions\n"
        log_msg += "   " + "-" * 60 + "\n"

        self.log_message(log_msg)

        # Add thumbnail to preview
        self.add_process_thumbnail(file_path)

    def add_process_thumbnail(self, file_path: str):
        """Add thumbnail to processing preview"""
        try:
            # Load and resize image
            image = Image.open(file_path)

            # Get original dimensions
            orig_width, orig_height = image.size

            image.thumbnail((120, 120), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Create thumbnail frame with border
            thumb_frame = ttk.Frame(self.process_thumbnails_frame, relief="solid", borderwidth=1)
            thumb_frame.pack(side="left", padx=5, pady=5)

            # Display image
            image_label = ttk.Label(thumb_frame, image=photo)
            image_label.pack()

            # Get file size
            file_size = os.path.getsize(file_path) / 1024  # KB

            # Display info below image
            filename = os.path.basename(file_path)
            info_text = f"{filename}\n{orig_width}x{orig_height} | {file_size:.0f}KB"

            name_label = ttk.Label(thumb_frame, text=info_text,
                                  wraplength=120, font=('TkDefaultFont', 7),
                                  justify="center")
            name_label.pack()

            # Keep references
            self.process_thumbnail_refs.append(photo)
            self.process_thumbnails.append(thumb_frame)

            # Limit to last 20 thumbnails
            if len(self.process_thumbnails) > 20:
                old_frame = self.process_thumbnails.pop(0)
                old_frame.destroy()
                self.process_thumbnail_refs.pop(0)

            # Update scroll region
            self.process_thumbnails_frame.update_idletasks()
            self.process_canvas.configure(scrollregion=self.process_canvas.bbox("all"))
            # Auto-scroll to the right to show latest
            self.process_canvas.xview_moveto(1.0)

        except Exception as e:
            logger.error(f"Error adding process thumbnail: {e}")

    # ============================================================================
    # SEARCH METHODS
    # ============================================================================

    def search_faces(self):
        """Search for similar faces with metadata/hybrid options"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        # Check for model mismatch before searching
        mismatch_info = self.system.db_manager.check_embedding_model_mismatch(
            self.system.config.embedding_model
        )

        if mismatch_info['has_mismatch'] and mismatch_info['total_count'] > 0:
            warning = (
                f"⚠️ CANNOT SEARCH - MODEL MISMATCH!\n\n"
                f"Current model: {mismatch_info['current_model']}\n"
                f"Database contains embeddings from different models:\n"
            )
            for model, count in mismatch_info['models_found'].items():
                warning += f"  • {model}: {count} embeddings\n"

            warning += (
                f"\nSearching with mismatched models produces INCORRECT results.\n\n"
                f"Please click 'Re-embed All Data' button in Configuration tab\n"
                f"to update all embeddings with '{mismatch_info['current_model']}' model."
            )

            messagebox.showerror("Model Mismatch - Cannot Search", warning)
            self.log_message("Search blocked due to model mismatch", "error")
            return

        search_mode = self.search_mode_var.get()

        try:
            # Build metadata filter
            metadata_filter = self._build_metadata_filter()

            # Perform search based on mode
            if search_mode == "metadata":
                # Metadata-only search
                if not metadata_filter:
                    messagebox.showwarning("Warning", "Please select at least one metadata filter for metadata search")
                    return
                results = self.system.db_manager.search_by_metadata(metadata_filter, self.num_results_var.get())
                self.log_message(f"Metadata search with filters: {metadata_filter}")

            elif search_mode == "vector" or search_mode == "hybrid":
                # Vector or hybrid search
                image_path = self.search_image_var.get()
                if not image_path or not os.path.exists(image_path):
                    messagebox.showerror("Error", "Please select a valid image file for vector/hybrid search")
                    return

                # Create embedding for search image using configured model
                analyzer = FaceAnalyzer()
                embedder = FaceEmbedder(model_name=self.system.config.embedding_model)

                features = analyzer.analyze_face(image_path)
                embedding = embedder.create_embedding(image_path, features)

                if search_mode == "hybrid" and metadata_filter:
                    # Hybrid search - vector + metadata
                    results = self.system.db_manager.hybrid_search(embedding, metadata_filter, self.num_results_var.get())
                    self.log_message(f"Hybrid search with filters: {metadata_filter}")
                else:
                    # Vector-only search
                    results = self.system.db_manager.search_faces(embedding, self.num_results_var.get())
                    self.log_message("Vector similarity search")

            else:
                messagebox.showerror("Error", f"Unknown search mode: {search_mode}")
                return

            # Display results
            self.display_search_results(results)
            self.system.stats.increment_search_queries()

            self.log_message(f"Search completed: {len(results)} results found")

        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")
            self.log_message(f"Search error: {e}", "error")

    def _build_metadata_filter(self) -> Dict[str, Any]:
        """Build metadata filter from GUI selections"""
        metadata_filter = {}

        # Demographic filters
        if self.sex_filter_var.get() != "any":
            metadata_filter['sex'] = self.sex_filter_var.get()

        if self.age_filter_var.get() != "any":
            metadata_filter['age_group'] = self.age_filter_var.get()

        if self.skin_tone_filter_var.get() != "any":
            metadata_filter['skin_tone'] = self.skin_tone_filter_var.get()

        if self.skin_color_filter_var.get() != "any":
            metadata_filter['skin_color'] = self.skin_color_filter_var.get()

        if self.hair_color_filter_var.get() != "any":
            metadata_filter['hair_color'] = self.hair_color_filter_var.get()

        # Image property filters
        if self.brightness_filter_var.get() != "any":
            metadata_filter['brightness_level'] = self.brightness_filter_var.get()

        if self.quality_filter_var.get() != "any":
            metadata_filter['image_quality'] = self.quality_filter_var.get()

        # Face detection filter
        if self.has_face_var.get() == "yes":
            metadata_filter['has_face'] = True
        elif self.has_face_var.get() == "no":
            metadata_filter['has_face'] = False

        return metadata_filter

    def display_search_results(self, results: List[Dict[str, Any]]):
        """Display search results"""
        # Clear previous results
        for widget in self.results_frame_inner.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.results_frame_inner, text="No results found").pack(pady=20)
            return

        # Display results
        for i, result in enumerate(results):
            result_frame = ttk.Frame(self.results_frame_inner)
            result_frame.pack(fill="x", padx=5, pady=5)

            # Result info
            info_text = f"Result {i+1}: Distance: {result['distance']:.3f}\nPath: {result['metadata'].get('file_path', 'Unknown')}"
            ttk.Label(result_frame, text=info_text).pack(side="left")

            # Try to display image thumbnail
            try:
                image_path = result['metadata'].get('file_path')
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path)
                    image.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    image_label = ttk.Label(result_frame, image=photo)
                    image_label.image = photo  # Keep a reference
                    image_label.pack(side="right")
            except Exception:
                pass  # Skip image display if error

        # Update scroll region
        self.results_frame_inner.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    # ============================================================================
    # UTILITY AND HELPER METHODS
    # ============================================================================

    def browse_faces_dir(self):
        """Browse for faces directory"""
        directory = filedialog.askdirectory(initialdir=self.faces_dir_var.get())
        if directory:
            self.faces_dir_var.set(directory)

    def browse_search_image(self):
        """Browse for search image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.search_image_var.set(file_path)
            self.update_query_preview(file_path)

    def update_query_preview(self, image_path: str):
        """Update the query image preview"""
        try:
            if not os.path.exists(image_path):
                self.query_preview_label.config(text="Image not found", image='')
                self.query_preview_photo = None
                return

            # Load and resize image for preview
            image = Image.open(image_path)

            # Calculate resize to fit in preview (max 250x250)
            max_size = 250
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update label
            self.query_preview_label.config(image=photo, text='')
            self.query_preview_photo = photo  # Keep reference

            self.log_message(f"Query image preview updated: {os.path.basename(image_path)}")

        except Exception as e:
            self.log_message(f"Error updating query preview: {e}", "error")
            self.query_preview_label.config(text="Error loading image", image='')
            self.query_preview_photo = None

    def test_pg_connection(self):
        """Test PostgreSQL connection"""
        try:
            from pgvector_db import PgVectorDatabaseManager
            from core import SystemConfig

            # Create temporary config
            config = SystemConfig()
            config.db_type = "pgvector"
            config.db_host = self.pg_host_var.get()
            config.db_port = int(self.pg_port_var.get())
            config.db_name = self.pg_db_var.get()
            config.db_user = self.pg_user_var.get()
            config.db_password = self.pg_password_var.get()

            # Try to connect
            db_manager = PgVectorDatabaseManager(config)
            if db_manager.initialize():
                stats = db_manager.get_stats()
                db_manager.close()

                messagebox.showinfo(
                    "Connection Successful",
                    f"Successfully connected to PostgreSQL!\n\n"
                    f"Database: {config.db_name}\n"
                    f"Total faces: {stats.get('total_faces', 0)}\n"
                    f"Database size: {stats.get('database_size', 'Unknown')}"
                )
                self.log_message("PostgreSQL connection test successful")
            else:
                messagebox.showerror(
                    "Connection Failed",
                    "Failed to connect to PostgreSQL.\n\n"
                    "Please check:\n"
                    "- PostgreSQL is running\n"
                    "- Credentials are correct\n"
                    "- Database exists\n"
                    "- pgvector extension is enabled"
                )
                self.log_message("PostgreSQL connection test failed", "error")

        except Exception as e:
            messagebox.showerror(
                "Connection Error",
                f"Error testing PostgreSQL connection:\n\n{str(e)}"
            )
            self.log_message(f"PostgreSQL connection error: {e}", "error")

    def refresh_status(self):
        """Refresh system status"""
        self.log_message("Status refreshed")

    def reset_statistics(self):
        """Reset system statistics"""
        if self.system:
            self.system.stats = type(self.system.stats)()
            self.log_message("Statistics reset")

    def save_configuration(self):
        """Save current configuration"""
        try:
            if self.system:
                # Update system configuration from GUI
                self.system.config.faces_dir = self.faces_dir_var.get()

                # Save PostgreSQL settings
                self.system.config.db_host = self.pg_host_var.get()
                self.system.config.db_port = int(self.pg_port_var.get())
                self.system.config.db_name = self.pg_db_var.get()
                self.system.config.db_user = self.pg_user_var.get()
                self.system.config.db_password = self.pg_password_var.get()

                self.system.config.download_delay = self.download_delay_var.get()
                self.system.config.batch_size = self.batch_size_var.get()
                self.system.config.max_workers = self.max_workers_var.get()
                self.system.config.embedding_model = self.embedding_model_var.get()
                self.system.config.download_source = self.download_source_var.get()

                # Save to file
                config_file = self.system.config.config_file
                self.system.config.save_to_file()

                # Show detailed save result
                db_info = f"PostgreSQL: {self.system.config.db_host}:{self.system.config.db_port}/{self.system.config.db_name}"

                save_summary = (
                    f"Configuration saved to: {config_file}\n\n"
                    f"Faces Directory: {self.system.config.faces_dir}\n"
                    f"Database: {db_info}\n"
                    f"Embedding Model: {self.system.config.embedding_model}\n"
                    f"Download Source: {self.system.config.download_source}\n"
                    f"Download Delay: {self.system.config.download_delay}s\n"
                    f"Batch Size: {self.system.config.batch_size}\n"
                    f"Max Workers: {self.system.config.max_workers}"
                )

                self.log_message(f"Configuration saved to {config_file}")
                messagebox.showinfo("Configuration Saved", save_summary)
            else:
                messagebox.showerror("Error", "System not initialized")
        except Exception as e:
            self.log_message(f"Error saving configuration: {e}", "error")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_configuration(self):
        """Load configuration from file"""
        try:
            config_file = "system_config.json"

            # Check if config file exists
            if not os.path.exists(config_file):
                messagebox.showwarning("Warning", f"Configuration file '{config_file}' not found. Using default settings.")
                return

            # Load configuration
            config = SystemConfig.from_file(config_file)

            if self.system:
                self.system.config = config
                self.update_configuration_from_system()

            # Show detailed load result
            db_info = f"PostgreSQL: {getattr(config, 'db_host', 'localhost')}:{getattr(config, 'db_port', 5432)}/{getattr(config, 'db_name', 'vector_db')}"

            load_summary = (
                f"Configuration loaded from: {config_file}\n\n"
                f"Faces Directory: {config.faces_dir}\n"
                f"Database: {db_info}\n"
                f"Embedding Model: {config.embedding_model}\n"
                f"Download Source: {config.download_source}\n"
                f"Download Delay: {config.download_delay}s\n"
                f"Batch Size: {config.batch_size}\n"
                f"Max Workers: {config.max_workers}"
            )

            self.log_message(f"Configuration loaded from {config_file}")
            messagebox.showinfo("Configuration Loaded", load_summary)

        except Exception as e:
            self.log_message(f"Error loading configuration: {e}", "error")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def check_embedding_models(self):
        """Check embedding model availability"""
        models_status = check_embedding_models()

        status_lines = ["Embedding Model Availability:\n"]

        install_commands = {
            'facenet': 'pip install facenet-pytorch torch torchvision',
            'arcface': 'pip install insightface onnxruntime',
            'deepface': 'pip install deepface',
            'vggface2': 'pip install deepface',
            'openface': 'pip install deepface'
        }

        for model, available in models_status.items():
            if available:
                status_lines.append(f"✓ {model.upper()}: Available")
            else:
                status_lines.append(f"✗ {model.upper()}: Not installed")
                if model in install_commands:
                    status_lines.append(f"  Install: {install_commands[model]}")

        # Update deps text
        self.deps_text.delete(1.0, tk.END)
        self.deps_text.insert(1.0, "\n".join(status_lines))

    def check_dependencies(self):
        """Check system dependencies"""
        deps_status = []

        # PostgreSQL/pgvector is always required
        deps_status.append("✓ PostgreSQL + pgvector: Required (see install.sh)")

        # Check OpenCV
        try:
            import cv2
            deps_status.append("✓ OpenCV: Available")
        except ImportError:
            deps_status.append("✗ OpenCV: Missing (pip install opencv-python)")

        # Check PIL
        try:
            from PIL import Image
            deps_status.append("✓ PIL/Pillow: Available")
        except ImportError:
            deps_status.append("✗ PIL/Pillow: Missing (pip install Pillow)")

        # Check other dependencies
        for module in ['numpy', 'requests']:
            try:
                __import__(module)
                deps_status.append(f"✓ {module}: Available")
            except ImportError:
                deps_status.append(f"✗ {module}: Missing (pip install {module})")

        # Update deps text
        self.deps_text.delete(1.0, tk.END)
        self.deps_text.insert(1.0, "\n".join(deps_status))

    def check_model_mismatch_on_startup(self):
        """Check for embedding model mismatches on startup"""
        if not self.system:
            return

        try:
            mismatch_info = self.system.db_manager.check_embedding_model_mismatch(
                self.system.config.embedding_model
            )

            if mismatch_info['has_mismatch'] and mismatch_info['total_count'] > 0:
                models_found = mismatch_info['models_found']
                current_model = mismatch_info['current_model']

                warning_msg = (
                    f"⚠️ EMBEDDING MODEL MISMATCH DETECTED!\n\n"
                    f"Current model: {current_model}\n"
                    f"Database contains:\n"
                )

                for model, count in models_found.items():
                    warning_msg += f"  • {model}: {count} embeddings\n"

                warning_msg += (
                    f"\nTotal embeddings: {mismatch_info['total_count']}\n\n"
                    f"⚠️ Searching with mismatched models will produce INCORRECT results!\n\n"
                    f"Recommended actions:\n"
                    f"1. Click 'Re-embed All Data' to update all embeddings with '{current_model}'\n"
                    f"2. Or change embedding model back to match database"
                )

                self.log_message(warning_msg, "error")
                self.model_warning_label.config(text=f"⚠️ Model Mismatch: {len(models_found)} different models in database!")

                # Show warning dialog
                response = messagebox.askquestion(
                    "Embedding Model Mismatch",
                    f"{warning_msg}\n\nDo you want to RE-EMBED ALL DATA now with '{current_model}'?",
                    icon='warning'
                )

                if response == 'yes':
                    self.reembed_all_data()
            else:
                self.model_warning_label.config(text="")

        except Exception as e:
            logger.error(f"Error checking model mismatch: {e}")

    def on_embedding_model_changed(self, *args):
        """Called when embedding model selection changes"""
        if not self.system:
            return

        new_model = self.embedding_model_var.get()
        current_db_model = self.system.config.embedding_model

        if new_model != current_db_model:
            # Check if database has data
            db_info = self.system.db_manager.get_collection_info()
            count = db_info.get('count', 0)

            if count > 0:
                self.model_warning_label.config(
                    text=f"⚠️ WARNING: Changing model will require re-embedding {count} faces!"
                )
            else:
                self.model_warning_label.config(text="")

    def reembed_all_data(self):
        """Re-embed all data with the current embedding model"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            # Get current database info
            db_info = self.system.db_manager.get_collection_info()
            count = db_info.get('count', 0)
            current_model = self.system.config.embedding_model

            if count == 0:
                messagebox.showinfo("Info", "No data in database to re-embed.")
                return

            # Confirm with user
            confirm_msg = (
                f"RE-EMBED ALL DATA\n\n"
                f"This will:\n"
                f"1. Clear all {count} existing embeddings from database\n"
                f"2. Re-process all face images in {self.system.config.faces_dir}\n"
                f"3. Create new embeddings using: {current_model}\n\n"
                f"This operation cannot be undone and may take several minutes.\n\n"
                f"Continue?"
            )

            response = messagebox.askokcancel("Confirm Re-embedding", confirm_msg, icon='warning')

            if not response:
                return

            self.log_message(f"Starting re-embedding with model: {current_model}")

            # Clear existing data
            if not self.system.db_manager.clear_all_data():
                messagebox.showerror("Error", "Failed to clear database")
                return

            self.log_message(f"Cleared {count} existing embeddings")

            # Update the processor with new model
            self.system.processor.embedder = FaceEmbedder(model_name=current_model)
            self.system.processor.processed_files.clear()

            # Start processing all faces
            self.log_message("Re-embedding all faces...")
            self.start_processing()

            # Clear warning
            self.model_warning_label.config(text="")

            messagebox.showinfo("Re-embedding Started",
                f"Re-embedding all faces with '{current_model}' model.\n\n"
                f"Check the 'Process & Embed' tab for progress."
            )

        except Exception as e:
            self.log_message(f"Error re-embedding data: {e}", "error")
            messagebox.showerror("Error", f"Failed to re-embed data: {e}")

    def run(self):
        """Run the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log_message("Application interrupted")
        finally:
            # Cleanup
            if self.system:
                if hasattr(self.system, 'downloader'):
                    self.system.downloader.stop_download_loop()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point

    Creates and runs the Face Processing GUI application.
    The application provides a complete workflow for:
    - Downloading AI-generated face images
    - Processing and embedding faces
    - Searching for similar faces
    - Managing system configuration
    """
    app = IntegratedFaceGUI()
    app.run()


if __name__ == "__main__":
    main()