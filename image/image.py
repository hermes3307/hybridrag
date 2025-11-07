#!/usr/bin/env python3
"""
Image Processing GUI Application

A comprehensive graphical interface for image processing including:
- Image downloading from AI generation services
- Image analysis
- Vector embedding generation
- Similarity search and metadata filtering

This application provides a complete workflow for building and querying
an image recognition database using PostgreSQL with pgvector and various embedding models.
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

# Lazy import for core backend functionality - will be loaded when needed
_core_loaded = False
IntegratedFaceSystem = None
SystemConfig = None
FaceAnalyzer = None
FaceEmbedder = None
AVAILABLE_MODELS = None
check_embedding_models = None

def _load_core_modules():
    """Lazy load heavy core modules only when needed"""
    global _core_loaded, IntegratedImageSystem, SystemConfig, ImageAnalyzer
    global ImageEmbedder, AVAILABLE_MODELS, check_embedding_models

    if not _core_loaded:
        try:
            from core import (
                IntegratedImageSystem as _IntegratedImageSystem,
                SystemConfig as _SystemConfig,
                ImageAnalyzer as _ImageAnalyzer,
                ImageEmbedder as _ImageEmbedder,
                AVAILABLE_MODELS as _AVAILABLE_MODELS,
                check_embedding_models as _check_embedding_models
            )
            IntegratedImageSystem = _IntegratedImageSystem
            SystemConfig = _SystemConfig
            ImageAnalyzer = _ImageAnalyzer
            ImageEmbedder = _ImageEmbedder
            AVAILABLE_MODELS = _AVAILABLE_MODELS
            check_embedding_models = _check_embedding_models
            _core_loaded = True
        except ImportError as e:
            print(f"Error importing core backend: {e}")
            print("Make sure core.py is in the same directory")
            sys.exit(1)

    return (IntegratedImageSystem, SystemConfig, ImageAnalyzer,
            ImageEmbedder, AVAILABLE_MODELS, check_embedding_models)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GUILogHandler(logging.Handler):
    """Custom logging handler that sends logs to the GUI"""
    def __init__(self, gui_callback):
        super().__init__()
        self.gui_callback = gui_callback

    def emit(self, record):
        """Send log record to GUI"""
        try:
            msg = self.format(record)
            # Extract just the message part (remove timestamp and level)
            # Format: "2025-11-02 10:59:25,137 - INFO - message"
            parts = msg.split(' - ', 2)
            if len(parts) >= 3:
                message = parts[2]  # Get just the message
            else:
                message = msg

            # Call GUI callback with the message
            self.gui_callback(message)
        except Exception:
            self.handleError(record)

class IntegratedImageGUI:
    """
    Main GUI Application for Image Processing System

    Provides a tabbed interface with the following features:
    1. System Overview - Monitor system status and statistics
    2. Download Images - Download AI-generated images or capture from camera
    3. Process & Embed - Create vector embeddings from images
    4. Search Images - Query similar images using vector similarity or metadata
    5. Configuration - Manage system settings and database
    """

    # Model name mapping: Display Name -> Internal Name
    MODEL_DISPLAY_TO_INTERNAL = {
        "Statistical": "statistical",
        "CLIP": "clip",
        "YOLO": "yolo",
        "ResNet": "resnet"
    }

    # Model name mapping: Internal Name -> Display Name
    MODEL_INTERNAL_TO_DISPLAY = {
        "statistical": "Statistical",
        "clip": "CLIP",
        "yolo": "YOLO",
        "resnet": "ResNet"
    }

    def __init__(self):
        """Initialize the GUI application"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Image Processing System")

        # Set window size to specific dimensions
        window_width = 960
        window_height = 650

        # Center the window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Core system components
        self.system = None  # IntegratedImageSystem instance
        self.download_thread = None  # Background download thread
        self.processing_thread = None  # Background processing thread

        # Application state
        self.is_downloading = False
        self.is_processing = False
        self.last_stats_update = 0

        # Create GUI widgets (tabs)
        self.create_widgets()

        # Pack the notebook immediately so tabs are visible
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create a centralized log frame
        main_log_frame = ttk.LabelFrame(self.root, text="System Log", padding=5)
        main_log_frame.pack(fill="both", expand=False, padx=5, pady=(5, 0))

        # Create the main log text widget
        self.main_log_text = scrolledtext.ScrolledText(main_log_frame, height=6)
        self.main_log_text.pack(fill="both", expand=False)


        # Show a loading message
        self.log_message("Loading system components in background...")

        # Set up logging handler to redirect all Python logger output to GUI
        self.setup_logging_handler()

        # Defer system initialization to run after GUI is shown
        # This makes the window appear much faster
        self.root.after(100, self.initialize_system_deferred)

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
        - Download Images: Acquire image data
        - Process & Embed: Generate vector embeddings
        - Search Images: Query the database
        - Configuration: System settings
        """
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)

        # Create the system menu
        self.create_system_menu()

        # Tab 1: System Overview - Status monitoring and statistics
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="System Overview")
        self.create_overview_tab()

        # Tab 2: Download - Image acquisition
        self.download_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.download_frame, text="Download Images")
        self.create_download_tab()

        # Tab 3: Process/Embed - Vector embedding generation
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text="Process & Embed")
        self.create_process_tab()

        # Tab 4: Search - Database queries
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Search Images")
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
            ("Total Images", "total_images"),
            ("Embedding Models", "embedding_models"),
            ("Download Rate", "download_rate"),
            ("Processing Rate", "process_rate"),
            ("System Uptime", "uptime")
        ]

        for i, (label, key) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", padx=(0, 10))
            if key == "embedding_models":
                # Embedding models label with colored text
                self.status_labels[key] = ttk.Label(status_frame, text="Loading...",
                                                   font=('TkDefaultFont', 9))
            else:
                self.status_labels[key] = ttk.Label(status_frame, text="Loading..." if i == 0 else "Waiting for system...")
            self.status_labels[key].grid(row=i, column=1, sticky="w")

        # Statistics frame
        stats_frame = ttk.LabelFrame(self.overview_frame, text="Statistics", padding=10)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Statistics text widget
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=6, width=70)
        self.stats_text.pack(fill="both", expand=True)



        # Control buttons
        control_frame = ttk.Frame(self.overview_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(control_frame, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Check PostgreSQL", command=self.check_postgresql_status).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Reset Statistics", command=self.reset_statistics).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Clear Log", command=self.clear_overview_log).pack(side="left", padx=5)

    def create_download_tab(self):
        """Create download images tab"""

        # Configure grid weights for proper resizing
        self.download_frame.columnconfigure(0, weight=1)
        self.download_frame.rowconfigure(3, weight=0)  # Preview frame expands

        # Download control frame
        control_frame = ttk.LabelFrame(self.download_frame, text="Download Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Download settings
        ttk.Label(control_frame, text="Download Source:").grid(row=0, column=0, sticky="w")
        self.download_source_var = tk.StringVar(value="picsum_landscape")
        # Available download sources from ImageDownloader
        source_options = [
            "picsum_general",
            "picsum_landscape",
            "picsum_square",
            "picsum_portrait",
            "picsum_hd"
        ]
        source_combo = ttk.Combobox(control_frame, textvariable=self.download_source_var,
                                   values=source_options, width=25, state="readonly")
        source_combo.grid(row=0, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Download Delay (seconds):").grid(row=1, column=0, sticky="w")
        self.download_delay_var = tk.DoubleVar(value=1.0)
        delay_spin = ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1,
                                textvariable=self.download_delay_var, width=10)
        delay_spin.grid(row=1, column=1, sticky="w", padx=(5, 0))

        ttk.Label(control_frame, text="Images Directory:").grid(row=2, column=0, sticky="w")
        self.images_dir_var = tk.StringVar(value="./images")
        ttk.Entry(control_frame, textvariable=self.images_dir_var, width=40).grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_images_dir).grid(row=2, column=2, padx=5)

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

        # Download preview frame - thumbnails with scrolling
        preview_frame = ttk.LabelFrame(self.download_frame, text="Downloaded Images Preview", padding=10)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # Create canvas with both horizontal and vertical scrollbars
        self.download_canvas = tk.Canvas(preview_frame, height=60)
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
        self.process_frame.rowconfigure(3, weight=0)  # Preview frame expands

        # Processing control frame
        control_frame = ttk.LabelFrame(self.process_frame, text="Processing Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Processing settings
        ttk.Label(control_frame, text="Embedding Models:").grid(row=0, column=0, sticky="w")
        self.process_embedding_model_label = ttk.Label(control_frame, text="Loading...")
        self.process_embedding_model_label.grid(row=0, column=1, columnspan=5, sticky="w", padx=(5, 0))

        # Batch Size and Max Workers in one row
        ttk.Label(control_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_size_var = tk.IntVar(value=50)
        batch_spin = ttk.Spinbox(control_frame, from_=1, to=200, increment=1,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=1, column=1, sticky="w", padx=(5, 10))

        ttk.Label(control_frame, text="Max Workers:").grid(row=1, column=2, sticky="w", padx=(10, 0))
        self.max_workers_var = tk.IntVar(value=4)
        workers_spin = ttk.Spinbox(control_frame, from_=1, to=8, increment=1,
                                  textvariable=self.max_workers_var, width=10)
        workers_spin.grid(row=1, column=3, sticky="w", padx=(5, 10))

        # Process All Images button on the right side of the same row
        self.process_button = ttk.Button(control_frame, text="Process All Images", command=self.start_processing)
        self.process_button.grid(row=1, column=4, sticky="w", padx=(10, 0))

        # Stop Processing button (conditionally shown)
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing)
        # Initially hidden

        # Processing buttons frame for other buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=5, pady=10)

        ttk.Button(button_frame, text="Process New Only", command=self.process_new_images).pack(side="left", padx=5)

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

        # Processing preview frame - thumbnails with scrolling
        process_preview_frame = ttk.LabelFrame(self.process_frame, text="Embedding Images Preview", padding=10)
        process_preview_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        process_preview_frame.columnconfigure(0, weight=1)
        process_preview_frame.rowconfigure(0, weight=1)

        # Create canvas with both horizontal and vertical scrollbars
        self.process_canvas = tk.Canvas(process_preview_frame, height=60)
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
        self.search_frame.columnconfigure(0, weight=2)  # Results/controls column
        self.search_frame.columnconfigure(1, weight=1)  # Preview column
        self.search_frame.rowconfigure(0, weight=0)      # Controls don't expand
        self.search_frame.rowconfigure(1, weight=0)      # Results expand vertically

        # Search control frame (left side)
        control_frame = ttk.LabelFrame(self.search_frame, text="Search Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Search by image
        ttk.Label(control_frame, text="Search by Image:").grid(row=0, column=0, sticky="w")
        self.search_image_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.search_image_var, width=20).grid(row=0, column=1, sticky="w", padx=(5, 5))
        ttk.Button(control_frame, text="Browse", command=self.browse_search_image).grid(row=0, column=2, sticky="w", padx=(0, 5))

        # Embedding model for search
        ttk.Label(control_frame, text="Search Model:").grid(row=1, column=0, sticky="w")
        self.search_embedding_model_var = tk.StringVar(value="CLIP")
        model_options = ["CLIP", "YOLO", "ResNet", "Statistical"]
        embedding_combo = ttk.Combobox(control_frame, textvariable=self.search_embedding_model_var,
                                      values=model_options, width=20, state="readonly")
        embedding_combo.grid(row=1, column=1, sticky="w", padx=(5, 0))


        # Number of results
        ttk.Label(control_frame, text="Number of Results:").grid(row=2, column=0, sticky="w")
        self.num_results_var = tk.IntVar(value=10)
        results_spin = ttk.Spinbox(control_frame, from_=1, to=50, increment=1,
                                  textvariable=self.num_results_var, width=10)
        results_spin.grid(row=2, column=1, sticky="w", padx=(5, 0))

        # Search mode selection
        ttk.Label(control_frame, text="Search Mode:").grid(row=2, column=0, sticky="w")
        self.search_mode_var = tk.StringVar(value="vector")
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=2, column=1, sticky="w", padx=(5, 0))
        ttk.Radiobutton(mode_frame, text="Vector (Image)", variable=self.search_mode_var, value="vector").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Metadata", variable=self.search_mode_var, value="metadata").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Hybrid", variable=self.search_mode_var, value="hybrid").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Mixed", variable=self.search_mode_var, value="mixed").pack(side="left", padx=5)

        # Metadata filter frame
        metadata_frame = ttk.LabelFrame(control_frame, text="Metadata Filters (Optional)", padding=5)
        metadata_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))



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
        ttk.Button(control_frame, text="Search Images", command=self.search_images).grid(row=4, column=0, columnspan=3, pady=10)

        # Query image preview frame (right side) - spans both rows
        query_preview_frame = ttk.LabelFrame(self.search_frame, text="Query Image Preview", padding=10)
        query_preview_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Preview label for query image
        self.query_preview_label = ttk.Label(query_preview_frame, text="No image selected",
                                             relief="solid", borderwidth=1,
                                             anchor="center", background="lightgray")
        self.query_preview_label.pack(fill="both", expand=True, padx=5, pady=5)
        self.query_preview_photo = None  # Keep reference to prevent garbage collection

        # Comparison preview frame (below query image)
        comparison_preview_frame = ttk.LabelFrame(query_preview_frame, text="Selected Result for Comparison", padding=5)
        comparison_preview_frame.pack(fill="both", expand=False, padx=5, pady=(10, 5))

        # Comparison image label
        self.comparison_preview_label = ttk.Label(comparison_preview_frame, text="Click a result to compare",
                                                  relief="solid", borderwidth=1,
                                                  anchor="center", background="lightgray")
        self.comparison_preview_label.pack(fill="both", expand=True, padx=5, pady=5)
        self.comparison_preview_photo = None  # Keep reference to prevent garbage collection

        # Comparison info label
        self.comparison_info_label = ttk.Label(comparison_preview_frame, text="",
                                               wraplength=240, justify="left")
        self.comparison_info_label.pack(fill="x", padx=5, pady=(0, 5))

        # Results frame (only left column, same width as controls)
        self.results_frame = ttk.LabelFrame(self.search_frame, text="Search Results", padding=10)
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Results display
        self.results_canvas = tk.Canvas(self.results_frame, height=60)
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
        self.pg_db_var = tk.StringVar(value="vector_images")
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
        self.embedding_model_var = tk.StringVar(value="CLIP")

        # Create dropdown with all models (display names)
        model_options = ["CLIP", "YOLO", "ResNet", "Statistical"]
        embedding_combo = ttk.Combobox(embedding_frame, textvariable=self.embedding_model_var,
                                      values=model_options, width=20, state="readonly")
        embedding_combo.grid(row=0, column=1, sticky="w", padx=(5, 10))

        ttk.Button(embedding_frame, text="Check Model Availability",
                  command=self.gui_check_embedding_models).grid(row=0, column=2, padx=5)

        # Bind model change event
        self.embedding_model_var.trace('w', self.on_embedding_model_changed)

        # Model change warning label
        self.model_warning_label = ttk.Label(embedding_frame, text="", foreground="red", font=('TkDefaultFont', 9, 'bold'))
        self.model_warning_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))



        # System config frame
        system_frame = ttk.LabelFrame(self.config_frame, text="System Configuration", padding=10)
        system_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)



        # Config buttons
        button_frame = ttk.Frame(self.config_frame)
        button_frame.grid(row=4, column=0, pady=10)

        ttk.Button(button_frame, text="Initialize Vector Database", command=self.initialize_vector_database).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Check Dependencies", command=self.check_dependencies).pack(side="left", padx=5)

    def setup_layout(self):
        """Setup the main layout"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        images_dir = self.images_dir_var.get()
        if not os.path.isdir(images_dir):
            messagebox.showerror("Error", f"Images directory not found: {images_dir}")
            return

        self.log_message("Starting metadata validation...")
        
        # Run validation in a separate thread to keep UI responsive
        threading.Thread(target=self._run_metadata_validation, daemon=True).start()

    def _run_metadata_validation(self):
        """The actual validation logic running in a background thread."""
        images_dir = self.images_dir_var.get()
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(images_dir).rglob(ext))

        missing_json = []
        invalid_json = []
        
        # Updated required keys for image recognition (not face recognition)
        required_keys = ['filename', 'image_id', 'md5_hash']

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
            self.log_message("✓ Metadata validation complete. All files are well-defined.", "success")
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
        self.log_message(f"✓ Metadata fix complete. Fixed: {fixed_count} files, Errors: {error_count}", "success" if error_count == 0 else "warning")

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

        faces_dir = self.images_dir_var.get()
        if not os.path.isdir(faces_dir):
            messagebox.showerror("Error", f"Faces directory not found: {faces_dir}")
            return

        self.log_message("Starting metadata validation...")
        
        # Run validation in a separate thread to keep UI responsive
        threading.Thread(target=self._run_metadata_validation, daemon=True).start()

    def _run_metadata_validation(self):
        """The actual validation logic running in a background thread."""
        faces_dir = self.images_dir_var.get()
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(faces_dir).rglob(ext))

        missing_json = []
        invalid_json = []
        
        # Updated required keys for image recognition (not face recognition)
        required_keys = ['filename', 'image_id', 'md5_hash']

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
            self.log_message("✓ Metadata validation complete. All files are well-defined.", "success")
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
        self.log_message(f"✓ Metadata fix complete. Fixed: {fixed_count} files, Errors: {error_count}", "success" if error_count == 0 else "warning")

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

    def _processing_progress(self, current, total, message):
        """Callback for embedding/processing progress - logs to panel"""
        def update_gui():
            try:
                if total > 0:
                    percentage = int((current / total) * 100)
                    status_msg = f"Processing: {current:,}/{total:,} files ({percentage}%)"
                    if message:
                        status_msg += f" - {message}"
                    self.log_message(status_msg, "info")
            except Exception as e:
                print(f"Error updating processing progress: {e}")

        self.root.after(0, update_gui)

    def _hash_loading_progress(self, count, total, message):
        """Callback for hash loading progress - logs to panel only"""
        # No GUI update, progress is logged in _hash_loading_complete
        pass

    def _hash_loading_complete(self, count, elapsed):
        """Callback when hash loading completes - logs to panel only"""
        def update_gui():
            try:
                # Log completion only
                self.log_message(f"✓ Duplicate detection ready ({count:,} image hashes loaded in {elapsed:.1f}s)", "success")
            except Exception as e:
                print(f"Error in hash loading complete: {e}")

        self.root.after(0, update_gui)

    def initialize_system_deferred(self):
        """Initialize the image processing system in background after UI is shown"""
        def init_worker():
            try:
                # Update window title to show loading
                self.root.after(0, lambda: self.root.title("Image Processing System - Loading..."))

                # Load core modules first (lazy loading)
                self.root.after(0, lambda: self.log_message("Starting system initialization..."))
                self.root.after(0, lambda: self.log_message("Loading core modules (this may take a moment)..."))
                _load_core_modules()
                self.root.after(0, lambda: self.log_message("✓ Core modules loaded successfully"))

                # Create and initialize system
                self.root.after(0, lambda: self.log_message("Initializing image processing system..."))
                self.root.after(0, lambda: self.log_message("Connecting to PostgreSQL database..."))
                self.system = IntegratedImageSystem()

                if self.system.initialize():
                    self.root.after(0, lambda: self.log_message("✓ Database connection established"))
                    self.root.after(0, lambda: self.log_message("✓ Image processor initialized"))
                    self.root.after(0, lambda: self.log_message("✓ System initialized successfully"))
                    # Update GUI in main thread
                    self.root.after(0, self.update_configuration_from_system)
                    self.root.after(0, self.update_available_models_label)
                    self.root.after(0, lambda: self.log_message("Checking embedding models..."))
                    # self.root.after(0, self.check_model_mismatch_on_startup)
                    # Restore window title
                    self.root.after(0, lambda: self.root.title("Image Processing System - Ready"))

                    # Now run setup_layout since system is initialized
                    self.root.after(0, self.setup_layout)

                    self.root.after(0, lambda: self.log_message("System is ready for use!"))

                    # Start background hash loading for duplicate detection
                    self.root.after(0, lambda: self.log_message("Starting background hash loading for duplicate detection..."))
                    self.system.downloader.start_background_hash_loading(
                        progress_callback=self._hash_loading_progress,
                        completion_callback=self._hash_loading_complete
                    )
                else:
                    self.log_message("✗ Failed to initialize system", "error")
                    self.root.after(0, lambda: self.root.title("Image Processing System - Error"))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", "Failed to initialize system. Check dependencies."))
            except Exception as e:
                self.log_message(f"✗ Error initializing system: {e}", "error")
                self.root.after(0, lambda: self.root.title("Image Processing System - Error"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Error initializing system: {e}"))

        # Run initialization in background thread
        threading.Thread(target=init_worker, daemon=True).start()

    def initialize_system(self):
        """Initialize the image processing system (synchronous version for re-initialization)"""
        try:
            # Ensure core modules are loaded
            _load_core_modules()

            self.system = IntegratedImageSystem()
            if self.system.initialize():
                self.log_message("System initialized successfully")
                self.update_configuration_from_system()
                # Check for model mismatch after initialization
                # self.check_model_mismatch_on_startup()
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
            self.images_dir_var.set(config.images_dir)

            # Load PostgreSQL database settings
            self.pg_host_var.set(getattr(config, 'db_host', 'localhost'))
            self.pg_port_var.set(str(getattr(config, 'db_port', 5432)))
            self.pg_db_var.set(getattr(config, 'db_name', 'vector_images'))
            self.pg_user_var.set(getattr(config, 'db_user', 'postgres'))
            self.pg_password_var.set(getattr(config, 'db_password', ''))

            self.download_delay_var.set(config.download_delay)
            self.batch_size_var.set(config.batch_size)
            self.max_workers_var.set(config.max_workers)
            # Convert internal model name to display name
            internal_model = config.embedding_model
            display_model = self.MODEL_INTERNAL_TO_DISPLAY.get(internal_model, "CLIP")
            self.embedding_model_var.set(display_model)

            # Update the Process & Embed tab label to show current model
            if hasattr(self, 'process_embedding_model_label'):
                self.process_embedding_model_label.config(text=display_model)

            self.download_source_var.set(config.download_source)

    def update_available_models_label(self):
        """Update the label in the Process & Embed tab with the list of available models."""
        if self.system:
            available_models = [self.MODEL_INTERNAL_TO_DISPLAY.get(model, model) for model, available in AVAILABLE_MODELS.items() if available]
            models_text = ", ".join(available_models)
            self.process_embedding_model_label.config(text=models_text)

    def setup_logging_handler(self):
        """Set up logging handler to redirect all logger output to GUI"""
        # Create a custom handler that sends logs to GUI
        gui_handler = GUILogHandler(self._log_from_external)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Also keep console handler for terminal output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # ONLY add to specific module loggers (not root logger to avoid duplicates)
        # Don't add to '__main__' to avoid loop with faces.py's log_message()
        for logger_name in ['pgvector_images', 'core', 'image_processor']:
            module_logger = logging.getLogger(logger_name)
            # Check if handler already added
            if not any(isinstance(h, GUILogHandler) for h in module_logger.handlers):
                module_logger.addHandler(gui_handler)
                module_logger.addHandler(console_handler)
                # Don't propagate to root logger to avoid duplicates
                module_logger.propagate = False

    def _log_from_external(self, message: str):
        """Callback for external logger to send messages to GUI (thread-safe)"""
        # Schedule the GUI update in the main thread
        self.root.after(0, lambda: self._add_to_gui_log(message))

    def _add_to_gui_log(self, message: str):
        """Add message to GUI log widgets (must run in main thread)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Log to the main log text widget
        if hasattr(self, 'main_log_text'):
            self.main_log_text.insert(tk.END, formatted_message)
            self.main_log_text.see(tk.END)

    def log_message(self, message: str, level: str = "info"):
        """Log message to appropriate text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        # Log to the main log text widget
        if hasattr(self, 'main_log_text'):
            self.main_log_text.insert(tk.END, formatted_message)
            self.main_log_text.see(tk.END)

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
        """Initialize vector database - recreate schema from scratch"""
        try:
            if not self.system:
                messagebox.showerror("Error", "System not initialized")
                return

            # Get current database info
            db_info = self.system.db_manager.get_collection_info()
            current_count = db_info.get('count', 0)

            # Show warning dialog
            warning_message = (
                "⚠️  WARNING: DESTRUCTIVE OPERATION ⚠️\n\n"
                "This will completely REINITIALIZE the vector database:\n\n"
                "• DROP all existing tables\n"
                "• DELETE all image embeddings\n"
                "• RECREATE schema from scratch\n"
                "• CREATE all indexes and functions\n\n"
                f"Current database contains {current_count} image records.\n\n"
                "This action CANNOT be undone!\n\n"
                "Do you want to proceed?"
            )

            # Ask for confirmation
            confirmed = messagebox.askyesno(
                "Confirm Database Reinitialization",
                warning_message,
                icon='warning'
            )

            if not confirmed:
                self.log_message("Database reinitialization cancelled by user")
                return

            # Second confirmation for safety
            final_confirm = messagebox.askyesno(
                "Final Confirmation",
                f"Are you ABSOLUTELY SURE you want to delete all {current_count} records?\n\n"
                "This is your last chance to cancel!",
                icon='warning'
            )

            if not final_confirm:
                self.log_message("Database reinitialization cancelled by user")
                return

            self.log_message("=" * 70)
            self.log_message("Starting database reinitialization...")
            self.log_message("=" * 70)

            # Perform reinitialization
            if self.system.db_manager.reinitialize_schema():
                self.log_message("✓ Database schema reinitialized successfully!")

                # Re-initialize the connection
                if self.system.db_manager.initialize():
                    self.log_message("✓ Database connection re-established")

                    messagebox.showinfo(
                        "Success",
                        "Vector database reinitialized successfully!\n\n"
                        "The database schema has been recreated with:\n"
                        "• Empty images table\n"
                        "• All indexes created\n"
                        "• All helper functions available\n\n"
                        "You can now start processing image data."
                    )
                else:
                    self.log_message("⚠️  Warning: Database reinitialized but connection failed", "warning")
                    messagebox.showwarning(
                        "Partial Success",
                        "Database schema was recreated, but connection verification failed.\n"
                        "Please check the logs and try reconnecting."
                    )
            else:
                self.log_message("✗ Failed to reinitialize database schema", "error")
                messagebox.showerror(
                    "Error",
                    "Failed to reinitialize database schema.\n\n"
                    "Please check:\n"
                    "• PostgreSQL is running\n"
                    "• Connection parameters are correct\n"
                    "• User has proper permissions\n"
                    "• schema.sql file exists\n\n"
                    "Check the logs for more details."
                )

        except Exception as e:
            self.log_message(f"✗ Error reinitializing vector database: {e}", "error")
            messagebox.showerror("Error", f"Error reinitializing vector database:\n\n{e}")



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
                    self.status_labels['total_images'].config(text=str(db_info.get('count', 0)))
                    self.status_labels['download_rate'].config(text=f"{stats.get('download_rate', 0):.2f}/sec")
                    self.status_labels['process_rate'].config(text=f"{stats.get('embed_rate', 0):.2f}/sec")
                    self.status_labels['uptime'].config(text=f"{stats.get('elapsed_time', 0):.0f}s")

                    # Update embedding model statistics
                    if 'embedding_models' in self.status_labels:
                        try:
                            model_info = self.system.db_manager.check_embedding_model_mismatch()
                            models_found = model_info.get('models_found', {})

                            if models_found:
                                # Desired order
                                model_order = ['clip', 'yolo', 'resnet', 'statistical']

                                # Create a list of model strings in the desired order
                                sorted_models = []
                                for model_name in model_order:
                                    if model_name in models_found:
                                        display_name = self.MODEL_INTERNAL_TO_DISPLAY.get(model_name, model_name)
                                        sorted_models.append(f"{display_name}: {models_found[model_name]}")

                                # Add any other models that might not be in the desired order list
                                for model, count in models_found.items():
                                    if model not in model_order:
                                        display_name = self.MODEL_INTERNAL_TO_DISPLAY.get(model, model)
                                        sorted_models.append(f"{display_name}: {count}")

                                model_text = ", ".join(sorted_models)
                                self.status_labels['embedding_models'].config(text=model_text)
                            else:
                                self.status_labels['embedding_models'].config(text="No embeddings yet")
                        except Exception as e:
                            self.status_labels['embedding_models'].config(text="Error loading")

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
            self.system.config.images_dir = self.images_dir_var.get()
            self.system.config.download_source = self.download_source_var.get()

            # Create faces directory
            os.makedirs(self.system.config.images_dir, exist_ok=True)

            # Start download
            self.download_thread = self.system.downloader.start_download_loop(
                callback=self.on_image_downloaded
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
        """Download a single image"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        try:
            file_path = self.system.downloader.download_image()
            if file_path:
                self.log_message(f"Downloaded: {os.path.basename(file_path)}")
            else:
                self.log_message("No new image downloaded (duplicate or error)")
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
                file_path = os.path.join(self.system.config.images_dir, filename)

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
                temp_dir = os.path.join(self.system.config.images_dir, "temp")
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

                messagebox.showinfo("Success", f"Picture captured for search!\n\nYou can now click 'Search Images' to find similar images.")

        except Exception as e:
            self.log_message(f"Camera capture error: {e}", "error")
            messagebox.showerror("Error", f"Failed to capture picture: {e}")

    def on_image_downloaded(self, file_path: str):
        """Callback when an image is downloaded"""
        # Use root.after to schedule GUI updates on the main thread
        self.root.after(0, lambda: self._update_download_image_ui(file_path))

    def _update_download_image_ui(self, file_path: str):
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
    # IMAGE PROCESSING METHODS
    # ============================================================================

    def start_processing(self):
        """Start processing all images"""
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
                all_files.extend(Path(self.system.config.images_dir).rglob(ext))
    
            new_files = self.system.processor.get_new_files_only()
            existing_count = len(all_files) - len(new_files)
    
            if len(all_files) == 0:
                messagebox.showinfo("No Files",
                    "No image files found in the images directory!\n\n"
                    f"Directory: {self.system.config.images_dir}")
                return
    
            # Show confirmation with details
            response = messagebox.askokcancel(
                "Process All Images",
                f"Process ALL images in directory:\n\n"
                f"Total files: {len(all_files)}\n"
                f"  • New files: {len(new_files)} (will be processed)\n"
                f"  • Already in DB: {existing_count} (will be skipped)\n\n"
                f"Using '{self.MODEL_INTERNAL_TO_DISPLAY.get(self.system.config.embedding_model, self.system.config.embedding_model)}' model\n\n"
                f"Duplicates will be automatically skipped.\n\n"
                f"Continue?"
            )
    
            if not response:
                return
    
        except Exception as e:
            self.log_message(f"Error checking files: {e}", "error")
    
        self.is_processing = True
        self.process_button.grid_remove()
        self.stop_button.grid(row=1, column=4, sticky="w", padx=(10, 0))

        def process_worker():
            try:
                self.system.processor.process_all_images(
                    callback=self.on_image_processed,
                    progress_callback=self._processing_progress
                )
                self.log_message("✓ Processing completed", "success")
            except Exception as e:
                self.log_message(f"✗ Processing error: {e}", "error")
            finally:
                self.is_processing = False
                self.root.after(0, lambda: self.stop_button.grid_remove())
                self.root.after(0, lambda: self.process_button.grid(row=1, column=4, sticky="w", padx=(10, 0)))
    
        self.processing_thread = threading.Thread(target=process_worker, daemon=True)
        self.processing_thread.start()
    
    def process_new_images(self):
        """Process only new images (not in database)"""
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
                    "All files in the images directory have already been processed!\n\n"
                    "No new images to embed.")
                return
    
            # Ask for confirmation
            response = messagebox.askokcancel(
                "Process New Files Only",
                f"Found {len(new_files)} NEW files that haven't been processed yet.\n\n"
                f"These files will be:\n"
                f"1. Analyzed for image features\n"
                f"2. Embedded using '{self.MODEL_INTERNAL_TO_DISPLAY.get(self.system.config.embedding_model, self.system.config.embedding_model)}' model\n"
                f"3. Added to the database\n\n"
                f"Already processed files will be skipped.\n\n"
                f"Continue?"
            )
    
            if not response:
                return
    
            self.is_processing = True
            self.process_button.grid_remove()
            self.stop_button.grid(row=1, column=4, sticky="w", padx=(10, 0))

            def process_worker():
                try:
                    self.log_message(f"Processing {len(new_files)} new files only...")
                    result_stats = self.system.processor.process_new_images_only(
                        callback=self.on_image_processed,
                        progress_callback=self._processing_progress
                    )
                    self.log_message(f"✓ New files processing completed: {result_stats['processed']} processed, {result_stats['errors']} errors", "success")
                except Exception as e:
                    self.log_message(f"✗ Processing error: {e}", "error")
                finally:
                    self.is_processing = False
                    self.root.after(0, lambda: self.stop_button.grid_remove())
                    self.root.after(0, lambda: self.process_button.grid(row=1, column=4, sticky="w", padx=(10, 0)))
    
            self.processing_thread = threading.Thread(target=process_worker, daemon=True)
            self.processing_thread.start()
    
        except Exception as e:
            self.log_message(f"Error checking new files: {e}", "error")
            messagebox.showerror("Error", f"Failed to check for new files: {e}")
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        self.stop_button.grid_remove()
        self.process_button.grid(row=1, column=4, sticky="w", padx=(10, 0))
        self.log_message("Processing stopped by user", "warning")
    
    def on_image_processed(self, image_data):
        """Callback when an image is processed"""
        # Use root.after to schedule GUI updates on the main thread
        self.root.after(0, lambda: self._update_process_image_ui(image_data))
    
    def _update_process_image_ui(self, image_data):
        """Update process UI on main thread"""
        # Display detailed file information
        file_path = image_data.file_path
        filename = os.path.basename(file_path)
        features = image_data.features
    
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
    
        log_msg += f"   🔑 Hash: {image_data.image_hash[:12]}...\n"

        # Note: In multi-model system, embeddings are stored separately per model
        # image_data.embedding is None - this is expected
        log_msg += f"   🧬 Embeddings: Created for all available models\n"
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

    def search_images(self):
        """Search for similar faces with metadata/hybrid options"""
        if not self.system:
            messagebox.showerror("Error", "System not initialized")
            return

        # Get the selected model from the dropdown
        display_model = self.search_embedding_model_var.get()
        model_name = self.MODEL_DISPLAY_TO_INTERNAL.get(display_model, "clip")

        # Check if the configured model has embeddings in the database
        mismatch_info = self.system.db_manager.check_embedding_model_mismatch(
            model_name
        )

        models_found = mismatch_info.get('models_found', {})

        # Check if the current model has any embeddings
        if models_found and model_name not in models_found:
            # Current model has NO embeddings in database
            warning = (
                f"⚠️ CANNOT SEARCH WITH {model_name.upper()}!\n\n"
                f"The selected model '{model_name}' has no embeddings in the database.\n\n"
                f"Database contains embeddings from:\n"
            )
            for model, count in models_found.items():
                warning += f"  • {model}: {count} embeddings\n"

            warning += (
                f"\nPlease either:\n"
                f"1. Change the search model to one of the above\n"
                f"2. Click 'Process New Only' to create {model_name} embeddings\n"
                f"3. Click 'Re-embed All Data' to recreate with {model_name}"
            )

            messagebox.showerror("No Embeddings for Selected Model", warning)
            self.log_message(f"Search blocked: No {model_name} embeddings found", "error")
            return

        # Log available models (informational only - multi-model system is normal)
        if len(models_found) > 1:
            self.log_message(f"Multi-model database detected. Searching with: {model_name}")

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
                analyzer = ImageAnalyzer()
                embedder = ImageEmbedder(model_name=model_name)

                features = analyzer.analyze_image(image_path)
                embedding = embedder.create_embedding(image_path, features)

                # Normalize embedding to 512 dimensions for search
                if len(embedding) != 512:
                    if len(embedding) < 512:
                        embedding.extend([0.0] * (512 - len(embedding)))
                    else:
                        embedding = embedding[:512]

                if search_mode == "hybrid" and metadata_filter:
                    # Hybrid search - vector + metadata
                    results = self.system.db_manager.hybrid_search(
                        embedding,
                        model_name,
                        metadata_filter,
                        self.num_results_var.get()
                    )
                    self.log_message(f"Hybrid search with {model_name} + filters: {metadata_filter}")
                else:
                    # Vector-only search with configured model
                    results = self.system.db_manager.search_images(
                        embedding,
                        model_name,
                        self.num_results_var.get()
                    )
                    self.log_message(f"Vector similarity search using {model_name} embeddings")

            elif search_mode == "mixed":
                image_path = self.search_image_var.get()
                if not image_path or not os.path.exists(image_path):
                    messagebox.showerror("Error", "Please select a valid image file for mixed search")
                    return

                # Create embeddings for each model
                clip_embedder = ImageEmbedder(model_name="clip")
                yolo_embedder = ImageEmbedder(model_name="yolo")
                resnet_embedder = ImageEmbedder(model_name="resnet")

                analyzer = ImageAnalyzer()
                features = analyzer.analyze_image(image_path)

                clip_embedding = clip_embedder.create_embedding(image_path, features)
                yolo_embedding = yolo_embedder.create_embedding(image_path, features)
                resnet_embedding = resnet_embedder.create_embedding(image_path, features)

                results = self.system.db_manager.multi_embedding_search(clip_embedding, yolo_embedding, resnet_embedding, self.num_results_var.get())
                self.log_message("Mixed search")

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
        """Display search results - images only"""
        # Clear previous results
        for widget in self.results_frame_inner.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.results_frame_inner, text="No results found").pack(pady=20)
            # Clear comparison preview
            self.comparison_preview_label.config(text="Click a result to compare", image='')
            self.comparison_preview_photo = None
            self.comparison_info_label.config(text="")
            return

        # Create a container frame for grid layout
        container = ttk.Frame(self.results_frame_inner)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # Display results as image thumbnails only
        for i, result in enumerate(results):
            try:
                image_path = result.get('file_path')
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path)
                    image.thumbnail((80, 80), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)

                    # Create clickable image label
                    image_label = tk.Label(container, image=photo, cursor="hand2",
                                          relief="raised", borderwidth=2, bg="white")
                    image_label.image = photo  # Keep a reference

                    # Grid layout - 5 images per row
                    row = i // 5
                    col = i % 5
                    image_label.grid(row=row, column=col, padx=3, pady=3)

                    # Bind click event to show popup with info
                    image_label.bind("<Button-1>", lambda e, r=result, idx=i+1: self.show_result_popup(r, idx))

                    # Show first result in comparison by default
                    if i == 0:
                        self.show_comparison(result, 1)
            except Exception as e:
                print(f"Error displaying result {i}: {e}")
                pass  # Skip image display if error

        # Update scroll region
        self.results_frame_inner.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def show_result_popup(self, result: Dict[str, Any], result_number: int):
        """Show popup with result information when image is clicked"""
        # Also update the comparison preview
        self.show_comparison(result, result_number)

        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"Result #{result_number} Details")
        popup.geometry("400x300")

        # Center the popup
        popup.transient(self.root)
        popup.grab_set()

        # Result information
        info_frame = ttk.Frame(popup, padding=10)
        info_frame.pack(fill="both", expand=True)

        distance = result.get('distance', 0.0)
        distance_str = "N/A" if distance == 0.0 else f"{distance:.4f}"
        image_path = result.get('file_path', 'Unknown')
        metadata = result.get('metadata', {})

        # Display info
        ttk.Label(info_frame, text=f"Result #{result_number}", font=('TkDefaultFont', 12, 'bold')).pack(anchor="w", pady=5)
        ttk.Label(info_frame, text=f"Distance: {distance_str}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"File: {os.path.basename(image_path)}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"Path: {image_path}", wraplength=350).pack(anchor="w", pady=2)

        # Show metadata if available
        if metadata:
            ttk.Separator(info_frame, orient='horizontal').pack(fill='x', pady=10)
            ttk.Label(info_frame, text="Metadata:", font=('TkDefaultFont', 10, 'bold')).pack(anchor="w", pady=5)

            for key, value in metadata.items():
                if key not in ['embedding', 'md5_hash']:  # Skip large/technical fields
                    ttk.Label(info_frame, text=f"{key}: {value}").pack(anchor="w", pady=1)

        # Close button
        ttk.Button(info_frame, text="Close", command=popup.destroy).pack(pady=10)

    def show_comparison(self, result: Dict[str, Any], result_number: int):
        """Show selected result in comparison preview under query image"""
        try:
            image_path = result.get('file_path')
            if not image_path or not os.path.exists(image_path):
                self.comparison_preview_label.config(text="Image not found", image='')
                self.comparison_preview_photo = None
                self.comparison_info_label.config(text="")
                return

            # Load and resize image for comparison preview (max 200x200)
            image = Image.open(image_path)
            max_size = 200
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Update comparison preview label
            self.comparison_preview_label.config(image=photo, text='')
            self.comparison_preview_photo = photo  # Keep reference

            # Update info label with result details
            distance = result.get('distance', 0.0)
            distance_str = "N/A" if distance == 0.0 else f"{distance:.4f}"
            metadata = result['metadata']
            info_text = (
                f"Result #{result_number}\n"
                f"Distance: {distance_str}\n"
                f"File: {os.path.basename(image_path)}"
            )

            # Add metadata if available - use correct field names
            if 'estimated_sex' in metadata:
                info_text += f"\nSex: {metadata['estimated_sex']}"
            if 'age_group' in metadata:
                info_text += f"\nAge: {metadata['age_group']}"
            if 'skin_tone' in metadata:
                info_text += f"\nSkin Tone: {metadata['skin_tone']}"

            self.comparison_info_label.config(text=info_text)

            self.log_message(f"Comparison preview updated: Result #{result_number}")

        except Exception as e:
            self.log_message(f"Error updating comparison preview: {e}", "error")
            self.comparison_preview_label.config(text="Error loading image", image='')
            self.comparison_preview_photo = None
            self.comparison_info_label.config(text="")

    # ============================================================================
    # UTILITY AND HELPER METHODS
    # ============================================================================

    def browse_images_dir(self):
        """Browse for images directory"""
        directory = filedialog.askdirectory(initialdir=self.images_dir_var.get())
        if directory:
            self.images_dir_var.set(directory)

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
            from core import PgVectorDatabaseManager, SystemConfig

            # Create temporary config
            config = SystemConfig()
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
                    f"Total images: {stats.get('total_images', 0)}\n"
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

    def check_postgresql_status(self):
        """Check PostgreSQL connection and system status"""
        import subprocess

        self.log_message("=== PostgreSQL System Check ===")

        # Check if PostgreSQL service is running
        try:
            result = subprocess.run(['systemctl', 'is-active', 'postgresql'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.log_message("✓ PostgreSQL service is running")
            else:
                self.log_message("✗ PostgreSQL service is not running")
                self.log_message("  Try: sudo systemctl start postgresql")
        except Exception as e:
            self.log_message(f"  Could not check service status: {e}")

        # Check database connection if system is initialized
        if self.system and self.system.db_manager:
            try:
                self.log_message("Checking database connection...")
                conn = self.system.db_manager.get_connection()
                if conn:
                    cursor = conn.cursor()

                    # Get PostgreSQL version
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    self.log_message(f"✓ Connected to: {version.split(',')[0]}")

                    # Check pgvector extension
                    cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    result = cursor.fetchone()
                    if result:
                        self.log_message(f"✓ pgvector extension version: {result[0]}")
                    else:
                        self.log_message("✗ pgvector extension not installed")

                    # Check images table
                    cursor.execute("SELECT COUNT(*) FROM images")
                    count = cursor.fetchone()[0]
                    self.log_message(f"✓ Images table contains {count} records")

                    # Check connection pool
                    cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
                    connections = cursor.fetchone()[0]
                    self.log_message(f"✓ Active database connections: {connections}")

                    cursor.close()
                    self.system.db_manager.return_connection(conn)
                    self.log_message("✓ Database check completed successfully")
            except Exception as e:
                self.log_message(f"✗ Database connection error: {e}", "error")
                self.log_message("  Check connection parameters in Configuration tab")
        else:
            self.log_message("✗ System not initialized yet")

        self.log_message("=== End of PostgreSQL Check ===")

    def reset_statistics(self):
        """Reset system statistics"""
        if self.system:
            self.system.stats = type(self.system.stats)()
            self.log_message("Statistics reset")

    def clear_overview_log(self):
        """Clear the system log"""
        if hasattr(self, 'main_log_text'):
            self.main_log_text.delete(1.0, tk.END)
            self.log_message("Log cleared")





    def gui_check_embedding_models(self):
        """Check availability of embedding models and show in messagebox"""
        try:
            if not self.system:
                messagebox.showerror("Error", "System not initialized")
                return

            # Check model availability
            model_status = check_embedding_models()

            # Build report string
            report = """
Embedding Model Availability:

"""
            for model, available in model_status.items():
                report += f"• {self.MODEL_INTERNAL_TO_DISPLAY.get(model, model)}: {'Available' if available else 'Not Available'}\n"

            report += """

Model Descriptions:

• Statistical (Default): Simple statistical features. Always available. Fast but lower accuracy.
  Size: 512 dimensions

• CLIP: Deep learning model for image and text similarity.
  Install: pip install torch torchvision transformers
  Size: 512 dimensions

• YOLO: Object detection model. Creates a bag-of-objects embedding.
  Install: pip install torch torchvision ultralytics
  Size: 80 dimensions

• ResNet: For deep visual features
  Install: pip install torch torchvision
  Size: 2048 dimensions
"""

            # Show in messagebox
            messagebox.showinfo("Embedding Model Status", report)

        except Exception as e:
            messagebox.showerror("Error", f"Error checking models: {e}")

    def check_dependencies(self):
        """Check for required dependencies and show in messagebox"""
        try:
            if not self.system:
                messagebox.showerror("Error", "System not initialized")
                return

            # Get dependency status from system
            deps_status = self.system.check_dependencies()

            # Build report string
            report = """
Dependency Status:

"""
            for dep, status in deps_status.items():
                report += f"• {dep}: {'Installed' if status else 'Not Installed'}\n"

            # Show in messagebox
            messagebox.showinfo("Dependency Status", report)

        except Exception as e:
            messagebox.showerror("Error", f"Error checking dependencies: {e}")

    def check_model_mismatch_on_startup(self):
        """Check for embedding model mismatches on startup"""
        if not self.system:
            return

        try:
            mismatch_info = self.system.db_manager.check_embedding_model_mismatch(
                self.system.config.embedding_model
            )

                    # if mismatch_info['has_mismatch'] and mismatch_info['total_count'] > 0:
                    #     models_found = mismatch_info['models_found']
                    #     current_model = mismatch_info['current_model']
            
                    #     warning_msg = (
                    #         f"⚠️ EMBEDDING MODEL MISMATCH DETECTED!\n\n"
                    #         f"Current model: {current_model}\n"
                    #         f"Database contains:\n"
                    #     )
            
                    #     for model, count in models_found.items():
                    #         warning_msg += f"  • {model}: {count} embeddings\n"
            
                    #     warning_msg += (
                    #         f"\nTotal embeddings: {mismatch_info['total_count']}\n\n"
                    #         f"⚠️ Searching with mismatched models will produce INCORRECT results!\n\n"
                    #         f"Recommended actions:\n"
                    #         f"1. Click 'Re-embed All Data' to update all embeddings with '{current_model}'\n"
                    #         f"2. Or change embedding model back to match database"
                    #     )
            
                    #     self.log_message(warning_msg, "error")
                    #     self.model_warning_label.config(text=f"⚠️ Model Mismatch: {len(models_found)} different models in database!")
            
                    #     # Show warning dialog
                    #     response = messagebox.askquestion(
                    #         "Embedding Model Mismatch",
                    #         f"{warning_msg}\n\nDo you want to RE-EMBED ALL DATA now with '{current_model}'?",
                    #         icon='warning'
                    #     )
            
                    #     if response == 'yes':
                    #         self.reembed_all_data()
                    # else:
                    #     self.model_warning_label.config(text="")
        except Exception as e:
            logger.error(f"Error checking model mismatch: {e}")

    def on_embedding_model_changed(self, *args):
        """Called when embedding model selection changes"""
        if not self.system:
            return

        # Convert display name to internal name
        display_model = self.embedding_model_var.get()
        new_model = self.MODEL_DISPLAY_TO_INTERNAL.get(display_model, "clip")
        current_db_model = self.system.config.embedding_model

        # Update the Process & Embed tab label
        if hasattr(self, 'process_embedding_model_label'):
            self.process_embedding_model_label.config(text=display_model)

        # Auto-save the configuration when model changes
        self.system.config.embedding_model = new_model
        try:
            self.system.config.save_to_file()
            self.log_message(f"Embedding model changed to {display_model} and saved")
        except Exception as e:
            self.log_message(f"Error saving config: {e}", "error")

        if new_model != current_db_model:
            # Check if database has data
            db_info = self.system.db_manager.get_collection_info()
            count = db_info.get('count', 0)

            if count > 0:
                self.model_warning_label.config(
                    text=f"⚠️ WARNING: Changing model will require re-embedding {count} images!"
                )
            else:
                self.model_warning_label.config(text="")






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

    Creates and runs the Image Processing GUI application.
    The application provides a complete workflow for:
    - Downloading AI-generated images
    - Processing and embedding images
    - Searching for similar images
    - Managing system configuration
    """
    app = IntegratedImageGUI()
    app.run()


if __name__ == "__main__":
    main()
