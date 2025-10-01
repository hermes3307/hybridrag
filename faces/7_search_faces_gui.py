#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Face Search Interface
Supports semantic search, metadata filtering, and combined search
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import io
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image, ImageTk
import tempfile
import subprocess

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # If already wrapped or not available, skip

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules with error handling
try:
    from face_database import FaceDatabase, FaceSearchInterface

    # Import directly from 3_collect_faces.py instead of the wrapper
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("collect_faces", os.path.join(os.path.dirname(__file__), "3_collect_faces.py"))
    collect_faces = module_from_spec(spec)
    spec.loader.exec_module(collect_faces)
    FaceAnalyzer = collect_faces.FaceAnalyzer
    FaceEmbedder = collect_faces.FaceEmbedder
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure face_database.py and 3_collect_faces.py are in the same directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedSearchGUI:
    """Unified search interface with semantic and metadata search capabilities"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 Unified Face Search Interface")
        self.root.geometry("1200x500")

        # Initialize components
        self.face_db = None
        self.search_interface = None
        self.analyzer = FaceAnalyzer()
        self.embedder = FaceEmbedder()

        # Single temp file for query image
        self.temp_query_file = None
        self.current_query_file = None
        self.search_results = []

        # Search mode
        self.search_mode = tk.StringVar(value="combined")

        # Setup UI and initialize database
        self.setup_ui()
        self.initialize_database()

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Set up the user interface"""
        # Main container with paned window for resizable sections
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Search controls
        left_frame = ttk.Frame(paned, width=400)
        paned.add(left_frame, weight=1)

        # Right panel - Results
        right_frame = ttk.Frame(paned, width=800)
        paned.add(right_frame, weight=2)

        # Setup left panel
        self.setup_search_controls(left_frame)

        # Setup right panel
        self.setup_results_panel(right_frame)

        # Status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_search_controls(self, parent):
        """Setup search controls panel"""
        # Title
        title_label = ttk.Label(parent, text="🔍 Search Controls",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Create notebook for organized tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.X, padx=5, pady=5)

        # Tab 1: Query Image
        self.setup_query_tab(notebook)

        # Search button right after query tab
        search_btn_frame = ttk.Frame(parent)
        search_btn_frame.pack(fill=tk.X, padx=5, pady=10)

        self.search_btn = ttk.Button(search_btn_frame, text="🔍 SEARCH",
                                     command=self.perform_search,
                                     style="Accent.TButton")
        self.search_btn.pack(fill=tk.X, ipady=10)

        # Clear button
        clear_btn = ttk.Button(search_btn_frame, text="🗑️ Clear All",
                              command=self.clear_all)
        clear_btn.pack(fill=tk.X, pady=(5, 0))

        # Tab 2: Search Mode
        self.setup_search_mode_tab(notebook)

        # Tab 3: Metadata Filters
        self.setup_filters_tab(notebook)

    def setup_query_tab(self, notebook):
        """Setup query image tab"""
        query_frame = ttk.Frame(notebook)
        notebook.add(query_frame, text="📷 Query Image")

        # Query image preview
        preview_frame = ttk.LabelFrame(query_frame, text="Query Image Preview")
        preview_frame.pack(fill=tk.X, padx=10, pady=10)

        self.query_canvas = tk.Canvas(preview_frame, width=120, height=120,
                                     bg='lightgray', highlightthickness=1,
                                     relief='ridge', bd=2)
        self.query_canvas.pack(pady=10)
        self.query_photo = None

        # Query info
        self.query_info_var = tk.StringVar(value="No query image selected")
        ttk.Label(preview_frame, textvariable=self.query_info_var,
                 wraplength=280).pack(pady=5)

        # Query source buttons
        source_frame = ttk.LabelFrame(query_frame, text="Query Image Source")
        source_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(source_frame, text="📁 Select from File",
                  command=self.select_query_file).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(source_frame, text="🌐 Download Random Face",
                  command=self.download_random_face).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(source_frame, text="📋 Paste from Clipboard",
                  command=self.paste_from_clipboard).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(source_frame, text="👁️ Preview Full Size",
                  command=self.preview_query_full).pack(fill=tk.X, padx=5, pady=2)

    def setup_search_mode_tab(self, notebook):
        """Setup search mode tab"""
        mode_frame = ttk.Frame(notebook)
        notebook.add(mode_frame, text="⚙️ Search Mode")

        # Search mode selection
        mode_select_frame = ttk.LabelFrame(mode_frame, text="Search Type")
        mode_select_frame.pack(fill=tk.X, padx=10, pady=10)

        modes = [
            ("combined", "🔄 Combined Search",
             "Use both semantic similarity and metadata filters"),
            ("semantic", "🧠 Semantic Search Only",
             "Search based on visual similarity only"),
            ("metadata", "📋 Metadata Search Only",
             "Search based on metadata filters only")
        ]

        for value, text, desc in modes:
            rb = ttk.Radiobutton(mode_select_frame, text=text,
                                variable=self.search_mode, value=value)
            rb.pack(anchor=tk.W, padx=5, pady=2)

            desc_label = ttk.Label(mode_select_frame, text=f"  {desc}",
                                  foreground="gray", font=("Arial", 8))
            desc_label.pack(anchor=tk.W, padx=20, pady=(0, 5))

        # Semantic search parameters
        semantic_frame = ttk.LabelFrame(mode_frame, text="Semantic Search Parameters")
        semantic_frame.pack(fill=tk.X, padx=10, pady=10)

        # Number of results
        ttk.Label(semantic_frame, text="Number of results:").pack(anchor=tk.W, padx=5, pady=2)
        self.num_results = tk.IntVar(value=10)
        ttk.Spinbox(semantic_frame, from_=1, to=50, textvariable=self.num_results,
                   width=10).pack(anchor=tk.W, padx=20, pady=2)

        # Similarity threshold
        ttk.Label(semantic_frame, text="Minimum similarity (%):").pack(anchor=tk.W, padx=5, pady=2)
        self.min_similarity = tk.DoubleVar(value=0.0)
        ttk.Scale(semantic_frame, from_=0, to=100, variable=self.min_similarity,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=2)

        self.similarity_label_var = tk.StringVar(value="0.0%")
        ttk.Label(semantic_frame, textvariable=self.similarity_label_var).pack(padx=20)

        # Update label when scale changes
        self.min_similarity.trace_add('write',
            lambda *args: self.similarity_label_var.set(f"{self.min_similarity.get():.1f}%"))

    def setup_filters_tab(self, notebook):
        """Setup metadata filters tab"""
        filters_frame = ttk.Frame(notebook)
        notebook.add(filters_frame, text="🔍 Filters")

        # Create canvas with scrollbar for filters
        canvas = tk.Canvas(filters_frame)
        scrollbar = ttk.Scrollbar(filters_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Age group filter
        age_frame = ttk.LabelFrame(scrollable_frame, text="Age Group")
        age_frame.pack(fill=tk.X, padx=5, pady=5)

        self.age_filter = tk.StringVar(value="any")
        for age in ["any", "young_adult", "adult", "mature_adult"]:
            ttk.Radiobutton(age_frame, text=age.replace("_", " ").title(),
                           variable=self.age_filter, value=age).pack(anchor=tk.W, padx=5)

        # Skin tone filter
        skin_frame = ttk.LabelFrame(scrollable_frame, text="Skin Tone")
        skin_frame.pack(fill=tk.X, padx=5, pady=5)

        self.skin_filter = tk.StringVar(value="any")
        for skin in ["any", "light", "medium", "dark"]:
            ttk.Radiobutton(skin_frame, text=skin.title(),
                           variable=self.skin_filter, value=skin).pack(anchor=tk.W, padx=5)

        # Quality filter
        quality_frame = ttk.LabelFrame(scrollable_frame, text="Image Quality")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)

        self.quality_filter = tk.StringVar(value="any")
        for quality in ["any", "high", "medium", "low"]:
            ttk.Radiobutton(quality_frame, text=quality.title(),
                           variable=self.quality_filter, value=quality).pack(anchor=tk.W, padx=5)

        # Brightness filter
        brightness_frame = ttk.LabelFrame(scrollable_frame, text="Brightness Range")
        brightness_frame.pack(fill=tk.X, padx=5, pady=5)

        self.use_brightness_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(brightness_frame, text="Enable brightness filter",
                       variable=self.use_brightness_filter).pack(anchor=tk.W, padx=5)

        ttk.Label(brightness_frame, text="Min brightness:").pack(anchor=tk.W, padx=5)
        self.min_brightness = tk.DoubleVar(value=0)
        ttk.Scale(brightness_frame, from_=0, to=255, variable=self.min_brightness,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

        ttk.Label(brightness_frame, text="Max brightness:").pack(anchor=tk.W, padx=5)
        self.max_brightness = tk.DoubleVar(value=255)
        ttk.Scale(brightness_frame, from_=0, to=255, variable=self.max_brightness,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

        # Date range filter
        date_frame = ttk.LabelFrame(scrollable_frame, text="Date Range")
        date_frame.pack(fill=tk.X, padx=5, pady=5)

        self.use_date_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(date_frame, text="Enable date filter",
                       variable=self.use_date_filter).pack(anchor=tk.W, padx=5)

        ttk.Label(date_frame, text="From date (YYYY-MM-DD):").pack(anchor=tk.W, padx=5)
        self.date_from = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.date_from).pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(date_frame, text="To date (YYYY-MM-DD):").pack(anchor=tk.W, padx=5)
        self.date_to = tk.StringVar()
        ttk.Entry(date_frame, textvariable=self.date_to).pack(fill=tk.X, padx=10, pady=2)

        # Reset filters button
        ttk.Button(scrollable_frame, text="🔄 Reset All Filters",
                  command=self.reset_filters).pack(fill=tk.X, padx=5, pady=10)

    def setup_results_panel(self, parent):
        """Setup results display panel"""
        # Title
        title_label = ttk.Label(parent, text="📊 Search Results",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Results summary
        self.results_summary_var = tk.StringVar(value="No search performed yet")
        ttk.Label(parent, textvariable=self.results_summary_var).pack(pady=5)

        # Results table with metadata
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create treeview with scrollbar
        columns = ("Rank", "Similarity", "Filename", "Age", "Skin", "Quality", "Brightness", "Download Date")
        self.results_tree = ttk.Treeview(table_frame, columns=columns,
                                        show="headings", height=12)

        # Configure columns
        self.results_tree.heading("Rank", text="#")
        self.results_tree.heading("Similarity", text="Match %")
        self.results_tree.heading("Filename", text="Filename")
        self.results_tree.heading("Age", text="Age")
        self.results_tree.heading("Skin", text="Skin")
        self.results_tree.heading("Quality", text="Quality")
        self.results_tree.heading("Brightness", text="Brightness")
        self.results_tree.heading("Download Date", text="Date")

        self.results_tree.column("Rank", width=40)
        self.results_tree.column("Similarity", width=80)
        self.results_tree.column("Filename", width=200)
        self.results_tree.column("Age", width=80)
        self.results_tree.column("Skin", width=70)
        self.results_tree.column("Quality", width=70)
        self.results_tree.column("Brightness", width=80)
        self.results_tree.column("Download Date", width=150)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical",
                                    command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal",
                                    command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set,
                                   xscrollcommand=h_scrollbar.set)

        # Pack tree and scrollbars
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Bind double-click to view image
        self.results_tree.bind("<Double-1>", self.view_result_image)

        # Results action buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="👁️ View Image",
                  command=self.view_selected_result).pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="📋 View Metadata",
                  command=self.view_result_metadata).pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="💾 Export Results",
                  command=self.export_results).pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="🗑️ Clear Results",
                  command=self.clear_results).pack(side=tk.LEFT, padx=2)

    def initialize_database(self):
        """Initialize the face database"""
        try:
            self.status_var.set("Initializing database...")
            self.face_db = FaceDatabase()
            self.search_interface = FaceSearchInterface(self.face_db)

            # Get database stats
            stats = self.face_db.get_database_stats()
            total_faces = stats.get('total_faces', 0)

            self.status_var.set(f"Ready - Database contains {total_faces:,} faces")
            logger.info(f"Database initialized with {total_faces} faces")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.status_var.set("Database initialization failed")
            messagebox.showerror("Database Error",
                               f"Failed to initialize database:\n{e}")

    def select_query_file(self):
        """Select query image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.set_query_image(file_path)

    def download_random_face(self):
        """Download a random face for query"""
        try:
            self.status_var.set("Downloading random face...")
            import requests

            response = requests.get("https://thispersondoesnotexist.com/", timeout=30)

            if response.status_code == 200:
                # Save to single temp file
                self.cleanup_temp_file()

                # Create single temp file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg',
                                                delete=False) as f:
                    f.write(response.content)
                    self.temp_query_file = f.name

                self.set_query_image(self.temp_query_file, from_download=True)
                self.status_var.set("Random face downloaded")
            else:
                messagebox.showerror("Download Failed",
                                   "Failed to download random face")

        except Exception as e:
            logger.error(f"Error downloading random face: {e}")
            messagebox.showerror("Error", f"Failed to download face:\n{e}")
            self.status_var.set("Download failed")

    def paste_from_clipboard(self):
        """Paste image from clipboard"""
        try:
            from PIL import ImageGrab

            image = ImageGrab.grabclipboard()

            if image is None:
                messagebox.showinfo("No Image",
                                  "No image found in clipboard")
                return

            # Save to single temp file
            self.cleanup_temp_file()

            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png',
                                            delete=False) as f:
                image.save(f, format='PNG')
                self.temp_query_file = f.name

            self.set_query_image(self.temp_query_file, from_clipboard=True)
            self.status_var.set("Image pasted from clipboard")

        except ImportError:
            messagebox.showerror("Not Supported",
                               "Clipboard paste requires PIL/Pillow library")
        except Exception as e:
            logger.error(f"Error pasting from clipboard: {e}")
            messagebox.showerror("Error", f"Failed to paste image:\n{e}")

    def set_query_image(self, file_path: str, from_download=False, from_clipboard=False):
        """Set and display query image"""
        try:
            self.current_query_file = file_path

            # Update preview
            self.update_query_preview()

            # Update info
            if from_download:
                info = "Random downloaded face"
            elif from_clipboard:
                info = "Image from clipboard"
            else:
                info = f"File: {os.path.basename(file_path)}"

            self.query_info_var.set(info)

        except Exception as e:
            logger.error(f"Error setting query image: {e}")
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def update_query_preview(self):
        """Update query image preview"""
        if not self.current_query_file or not os.path.exists(self.current_query_file):
            self.query_canvas.delete("all")
            self.query_canvas.configure(bg='lightgray')
            self.query_photo = None
            return

        try:
            # Load image
            pil_image = Image.open(self.current_query_file)

            # Convert to RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Resize to fit canvas
            max_size = 116
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            self.query_photo = ImageTk.PhotoImage(pil_image)

            # Display in canvas
            self.query_canvas.delete("all")
            self.query_canvas.configure(bg='white')
            self.query_canvas.create_image(60, 60, image=self.query_photo)

        except Exception as e:
            logger.error(f"Error updating query preview: {e}")
            self.query_canvas.delete("all")
            self.query_canvas.configure(bg='lightcoral')
            self.query_canvas.create_text(150, 150, text="Error loading image",
                                         fill="white")

    def preview_query_full(self):
        """Preview query image in full size"""
        if not self.current_query_file:
            messagebox.showinfo("No Image", "No query image selected")
            return

        if not os.path.exists(self.current_query_file):
            messagebox.showerror("Error", "Query image file not found")
            return

        try:
            if sys.platform == 'darwin':
                subprocess.run(['open', self.current_query_file])
            elif sys.platform == 'win32':
                os.startfile(self.current_query_file)
            else:
                subprocess.run(['xdg-open', self.current_query_file])
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def reset_filters(self):
        """Reset all metadata filters"""
        self.age_filter.set("any")
        self.skin_filter.set("any")
        self.quality_filter.set("any")
        self.use_brightness_filter.set(False)
        self.min_brightness.set(0)
        self.max_brightness.set(255)
        self.use_date_filter.set(False)
        self.date_from.set("")
        self.date_to.set("")
        self.status_var.set("Filters reset")

    def build_metadata_filter(self) -> Dict[str, Any]:
        """Build metadata filter dict from UI selections"""
        filters = {}

        # Age filter
        if self.age_filter.get() != "any":
            filters['estimated_age_group'] = self.age_filter.get()

        # Skin tone filter
        if self.skin_filter.get() != "any":
            filters['estimated_skin_tone'] = self.skin_filter.get()

        # Quality filter
        if self.quality_filter.get() != "any":
            filters['image_quality'] = self.quality_filter.get()

        # Brightness filter
        if self.use_brightness_filter.get():
            filters['mean_brightness'] = {
                'min': self.min_brightness.get(),
                'max': self.max_brightness.get()
            }

        # Date filter
        if self.use_date_filter.get() and (self.date_from.get() or self.date_to.get()):
            filters['download_date'] = {
                'from': self.date_from.get(),
                'to': self.date_to.get()
            }

        return filters

    def perform_search(self):
        """Perform search based on selected mode"""
        mode = self.search_mode.get()

        # Validate inputs based on mode
        if mode in ["semantic", "combined"]:
            if not self.current_query_file:
                messagebox.showwarning("No Query Image",
                                     "Please select a query image for semantic search")
                return

            if not os.path.exists(self.current_query_file):
                messagebox.showerror("Error", "Query image file not found")
                return

        try:
            self.status_var.set("Performing search...")
            self.clear_results()

            # Build metadata filters
            metadata_filters = self.build_metadata_filter()

            # Perform search based on mode
            if mode == "semantic":
                results = self._perform_semantic_search(metadata_filters)
            elif mode == "metadata":
                results = self._perform_metadata_search(metadata_filters)
            else:  # combined
                results = self._perform_combined_search(metadata_filters)

            # Display results
            self.display_results(results)

            self.status_var.set(f"Search completed - {len(results)} results found")

        except Exception as e:
            logger.error(f"Search error: {e}")
            messagebox.showerror("Search Error", f"Search failed:\n{e}")
            self.status_var.set("Search failed")

    def _perform_semantic_search(self, metadata_filters: Dict) -> List[Dict]:
        """Perform semantic similarity search"""
        num_results = self.num_results.get()

        # Use search interface
        raw_results = self.search_interface.search_by_image(
            self.current_query_file,
            num_results
        )

        if "error" in raw_results:
            raise Exception(raw_results["error"])

        # Process and filter results
        results = []
        search_data = raw_results.get("results", {})

        for i in range(search_data.get("count", 0)):
            metadata = search_data.get("metadatas", [])[i] if i < len(search_data.get("metadatas", [])) else {}
            distance = search_data.get("distances", [])[i] if i < len(search_data.get("distances", [])) else 1.0
            similarity = (1 - distance) * 100

            # Apply similarity threshold
            if similarity < self.min_similarity.get():
                continue

            # Apply metadata filters
            if not self._matches_filters(metadata, metadata_filters):
                continue

            result = {
                'rank': len(results) + 1,
                'face_id': search_data.get("ids", [])[i] if i < len(search_data.get("ids", [])) else "",
                'similarity': similarity,
                'metadata': metadata
            }
            results.append(result)

        return results

    def _perform_metadata_search(self, metadata_filters: Dict) -> List[Dict]:
        """Perform metadata-only search"""
        # Get all faces from database
        try:
            collection = self.face_db.collection
            all_results = collection.get(include=['metadatas'])

            results = []
            for i, metadata in enumerate(all_results.get('metadatas', [])):
                if self._matches_filters(metadata, metadata_filters):
                    result = {
                        'rank': len(results) + 1,
                        'face_id': all_results['ids'][i],
                        'similarity': None,
                        'metadata': metadata
                    }
                    results.append(result)

            # Limit results
            return results[:self.num_results.get()]

        except Exception as e:
            logger.error(f"Metadata search error: {e}")
            raise

    def _perform_combined_search(self, metadata_filters: Dict) -> List[Dict]:
        """Perform combined semantic + metadata search"""
        # First do semantic search with more results
        expanded_results = self.num_results.get() * 3

        # Temporarily increase num_results
        original_num = self.num_results.get()
        self.num_results.set(expanded_results)

        try:
            results = self._perform_semantic_search(metadata_filters)
            return results[:original_num]
        finally:
            self.num_results.set(original_num)

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches all filters"""
        if not filters:
            return True

        for key, value in filters.items():
            if key == 'mean_brightness' and isinstance(value, dict):
                brightness = metadata.get('mean_brightness', 0)
                if not (value['min'] <= brightness <= value['max']):
                    return False

            elif key == 'download_date' and isinstance(value, dict):
                date_str = metadata.get('download_date', '')
                if value.get('from') and date_str < value['from']:
                    return False
                if value.get('to') and date_str > value['to']:
                    return False

            else:
                if metadata.get(key) != value:
                    return False

        return True

    def display_results(self, results: List[Dict]):
        """Display search results in tree"""
        self.search_results = results

        if not results:
            self.results_summary_var.set("No results found")
            return

        for result in results:
            metadata = result['metadata']
            file_path = metadata.get('file_path', '')
            filename = os.path.basename(file_path) if file_path else result['face_id']

            # Extract metadata for display
            age = metadata.get('estimated_age_group', 'unknown')
            skin = metadata.get('estimated_skin_tone', 'unknown')
            quality = metadata.get('image_quality', 'unknown')
            brightness = metadata.get('mean_brightness', 0)

            # Get download date from source_metadata
            download_date = "N/A"
            source_meta = metadata.get('source_metadata', {})
            if isinstance(source_meta, str):
                try:
                    source_meta = eval(source_meta)
                except:
                    pass
            if isinstance(source_meta, dict):
                download_date = source_meta.get('download_date', 'N/A')

            # Format similarity
            similarity_str = f"{result['similarity']:.1f}%" if result['similarity'] is not None else "N/A"

            self.results_tree.insert("", tk.END, values=(
                result['rank'],
                similarity_str,
                filename,
                age,
                skin,
                quality,
                f"{brightness:.1f}",
                download_date
            ))

        mode_text = {"semantic": "semantic", "metadata": "metadata", "combined": "combined"}
        self.results_summary_var.set(
            f"Found {len(results)} results using {mode_text[self.search_mode.get()]} search"
        )

    def view_result_image(self, event):
        """View selected result image"""
        self.view_selected_result()

    def view_selected_result(self):
        """View selected result image"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a result to view")
            return

        item = self.results_tree.item(selection[0])
        rank = int(item['values'][0])

        result = self.search_results[rank - 1]
        file_path = result['metadata'].get('file_path', '')

        if file_path and os.path.exists(file_path):
            try:
                if sys.platform == 'darwin':
                    subprocess.run(['open', file_path])
                elif sys.platform == 'win32':
                    os.startfile(file_path)
                else:
                    subprocess.run(['xdg-open', file_path])
            except Exception as e:
                logger.error(f"Error opening image: {e}")
                messagebox.showerror("Error", f"Failed to open image:\n{e}")
        else:
            messagebox.showerror("Error", "Image file not found")

    def view_result_metadata(self):
        """View metadata for selected result"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a result to view metadata")
            return

        item = self.results_tree.item(selection[0])
        rank = int(item['values'][0])

        result = self.search_results[rank - 1]
        metadata = result['metadata']

        # Create metadata view window
        meta_window = tk.Toplevel(self.root)
        meta_window.title(f"Metadata - Result #{rank}")
        meta_window.geometry("600x500")

        # Metadata text
        text_widget = scrolledtext.ScrolledText(meta_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Format metadata
        text_widget.insert(tk.END, f"=== RESULT #{rank} METADATA ===\n\n")

        if result['similarity'] is not None:
            text_widget.insert(tk.END, f"Similarity: {result['similarity']:.2f}%\n\n")

        text_widget.insert(tk.END, json.dumps(metadata, indent=2))
        text_widget.config(state=tk.DISABLED)

    def export_results(self):
        """Export search results to JSON"""
        if not self.search_results:
            messagebox.showinfo("No Results", "No results to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                export_data = {
                    'search_mode': self.search_mode.get(),
                    'search_timestamp': datetime.now().isoformat(),
                    'query_image': self.current_query_file if self.current_query_file else None,
                    'num_results': len(self.search_results),
                    'filters': self.build_metadata_filter(),
                    'results': self.search_results
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success", f"Results exported to:\n{file_path}")

            except Exception as e:
                logger.error(f"Export error: {e}")
                messagebox.showerror("Error", f"Failed to export:\n{e}")

    def clear_results(self):
        """Clear search results"""
        self.search_results.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results_summary_var.set("No search performed yet")

    def clear_all(self):
        """Clear everything"""
        self.clear_results()
        self.current_query_file = None
        self.query_canvas.delete("all")
        self.query_canvas.configure(bg='lightgray')
        self.query_photo = None
        self.query_info_var.set("No query image selected")
        self.reset_filters()
        self.cleanup_temp_file()
        self.status_var.set("All cleared")

    def cleanup_temp_file(self):
        """Clean up temporary file"""
        if self.temp_query_file and os.path.exists(self.temp_query_file):
            try:
                os.remove(self.temp_query_file)
                logger.info(f"Cleaned up temp file: {self.temp_query_file}")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
        self.temp_query_file = None

    def on_closing(self):
        """Handle window closing"""
        self.cleanup_temp_file()
        self.root.destroy()

    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted")
        finally:
            self.cleanup_temp_file()

def main():
    """Main function"""
    print("🔍 Unified Face Search Interface")
    print("="*50)

    try:
        app = UnifiedSearchGUI()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()